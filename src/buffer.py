"""
Ring buffer and decode thread utilities for streaming video frames.

This stays UI-agnostic: it just moves frames from disk (via VideoReader)
into a bounded buffer that a consumer (e.g., GUI, logger, plotter) can pop.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from queue import Queue, Empty

import numpy as np

from video_io import VideoReader


@dataclass(frozen=True)
class FramePacket:
    """Single frame plus minimal timing metadata."""

    idx: int
    frame: np.ndarray
    wall_time: float  # seconds since epoch when captured/decoded


class FrameRingBuffer:
    """Fixed-size FIFO with optional blocking push/pop."""

    def __init__(self, capacity: int = 120):
        if capacity <= 0:
            raise ValueError("capacity must be > 0")
        self._buf: deque[FramePacket] = deque(maxlen=capacity)
        self._lock = threading.Lock()
        self._not_empty = threading.Condition(self._lock)
        self._not_full = threading.Condition(self._lock)
        self._stopped = False
        self._capacity = capacity

    # --------------------- properties --------------------- #
    @property
    def stopped(self) -> bool:
        return self._stopped

    @property
    def capacity(self) -> int:
        return self._capacity

    def __len__(self) -> int:
        return len(self._buf)

    def is_full(self) -> bool:
        return len(self._buf) >= self._capacity

    def clear(self) -> None:
        """Remove all buffered frames."""

        with self._lock:
            self._buf.clear()
            self._not_full.notify_all()

    # --------------------- operations --------------------- #
    def stop(self) -> None:
        """Signal consumers/producers to exit."""

        with self._lock:
            self._stopped = True
            self._not_empty.notify_all()
            self._not_full.notify_all()

    def push(self, packet: FramePacket, block: bool = False, timeout: Optional[float] = None) -> bool:
        """
        Insert a frame.

        If block=False (default), newest frames overwrite oldest when full.
        If block=True, waits until space is available or stop() is called.
        Returns True if stored, False if dropped or stop signaled.
        """

        with (self._not_full if block else self._lock):
            if block:
                ready = self._not_full.wait_for(lambda: not self.is_full() or self._stopped, timeout)
                if not ready or self._stopped:
                    return False
            self._buf.append(packet)  # deque(maxlen=cap) drops oldest when full
            self._not_empty.notify()
            return True

    def pop(self, block: bool = True, timeout: Optional[float] = None) -> Optional[FramePacket]:
        """
        Remove and return the oldest frame.
        Returns None on timeout or when stopped and empty.
        """

        with self._not_empty:
            if block:
                ready = self._not_empty.wait_for(lambda: len(self._buf) > 0 or self._stopped, timeout)
                if not ready or (self._stopped and len(self._buf) == 0):
                    return None
            elif len(self._buf) == 0:
                return None
            pkt = self._buf.popleft()
            self._not_full.notify()
            return pkt


def start_decoder(
    video_path: str | Path,
    buffer: FrameRingBuffer,
    start: int = 0,
    step: int = 1,
    block_on_full: bool = False,
) -> threading.Thread:
    """
    Launch a background thread that streams frames into the buffer.

    Args:
        video_path: path to the video file.
        buffer: FrameRingBuffer to fill.
        start: first frame index.
        step: frame stride (1 = every frame).
        block_on_full: if True, producer waits for space; if False, oldest frames
                       are dropped when capacity is reached (deque maxlen behavior).
    """

    def run() -> None:
        try:
            with VideoReader(video_path) as vr:
                for idx, frame in vr.iter_frames(start=start, step=step):
                    if buffer.stopped:
                        break
                    packet = FramePacket(idx=idx, frame=frame, wall_time=time.time())
                    stored = buffer.push(packet, block=block_on_full, timeout=0.5)
                    if not stored and not block_on_full:
                        # dropped due to full buffer; continue to latest frames
                        continue
                buffer.stop()
        except Exception:
            buffer.stop()
            raise

    t = threading.Thread(target=run, name="decoder-thread", daemon=True)
    t.start()
    return t


def start_decoder_controlled(
    video_path: str | Path,
    buffer: FrameRingBuffer,
    start: int = 0,
    step: int = 1,
    block_on_full: bool = False,
):
    """
    Decoder thread with a command queue for play/pause/seek/stop.

    Commands are dictionaries placed on the returned queue:
      - {'type': 'stop'}
      - {'type': 'pause', 'value': True|False}   # True pauses, False resumes
      - {'type': 'toggle_pause'}
      - {'type': 'play'}                         # resume
      - {'type': 'seek', 'frame': int}           # jump to absolute frame, clears buffer
      - {'type': 'step', 'frame': Optional[int]} # decode exactly one frame (at frame if provided)
    """

    commands: Queue = Queue()

    def run() -> None:
        playing = True
        last_idx = start
        at_end = False
        frame_interval = 0.0
        try:
            with VideoReader(video_path) as vr:
                vr.set_position(start)
                fps = vr.meta.fps
                frame_interval = 1.0 / fps if fps and fps > 0 else 0.0
                while not buffer.stopped:
                    # process any pending commands
                    while True:
                        try:
                            cmd = commands.get_nowait()
                        except Empty:
                            break
                        if not isinstance(cmd, dict) or "type" not in cmd:
                            continue
                        ctype = cmd["type"]
                        if ctype == "stop":
                            buffer.stop()
                            return
                        if ctype == "pause":
                            # value True => pause, False => play
                            playing = not bool(cmd.get("value", False))
                            at_end = False
                        elif ctype == "toggle_pause":
                            playing = not playing
                            at_end = False
                        elif ctype == "play":
                            playing = True
                            at_end = False
                        elif ctype == "seek":
                            target = int(cmd.get("frame", last_idx))
                            if target < 0:
                                target = 0
                            vr.set_position(target)
                            last_idx = target
                            buffer.clear()
                            at_end = False
                        elif ctype == "step":
                            target = cmd.get("frame", None)
                            if target is not None:
                                target = max(0, int(target))
                                vr.set_position(target)
                                last_idx = target
                            # read exactly one frame regardless of playing state
                            idx, frame = vr.read()
                            if frame is not None and idx >= 0:
                                last_idx = idx
                                packet = FramePacket(idx=idx, frame=frame, wall_time=time.time())
                                buffer.clear()
                                buffer.push(packet, block=False, timeout=0.0)
                            playing = False
                            at_end = False
                            continue

                    if not playing:
                        time.sleep(0.01)
                        continue

                    if block_on_full and buffer.is_full():
                        time.sleep(0.005)
                        continue

                    idx, frame = vr.read()
                    if frame is None or idx < 0:
                        # hit EOF; pause but keep thread alive for seeks
                        playing = False
                        at_end = True
                        time.sleep(0.01)
                        continue
                    last_idx = idx
                    if step > 1:
                        vr.set_position(idx + step)
                    packet = FramePacket(idx=idx, frame=frame, wall_time=time.time())
                    stored = buffer.push(packet, block=False, timeout=0.0)
                    if not stored and not block_on_full:
                        continue
                    if frame_interval > 0:
                        time.sleep(frame_interval)
        except Exception:
            buffer.stop()
            raise

    t = threading.Thread(target=run, name="decoder-thread-controlled", daemon=True)
    t.start()
    return commands, t


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Ring buffer smoke test")
    parser.add_argument("video", type=str, help="Path to video (e.g., test_video.mp4)")
    parser.add_argument("--capacity", type=int, default=16, help="Max frames to hold")
    parser.add_argument("--step", type=int, default=3, help="Frame stride for sampling")
    parser.add_argument("--block", action="store_true", help="Block producer when full instead of dropping oldest")
    args = parser.parse_args()

    buf = FrameRingBuffer(capacity=args.capacity)
    t = start_decoder(args.video, buf, step=args.step, block_on_full=args.block)

    count = 0
    while True:
        pkt = buf.pop(timeout=1.0)
        if pkt is None:
            break
        print(f"idx={pkt.idx:4d} shape={pkt.frame.shape} t={pkt.wall_time:.3f}")
        count += 1
    t.join()
    print(f"Consumed {count} frames; buffer stopped={buf.stopped}")
