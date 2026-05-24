"""
Lightweight, frame-accurate video access helpers built on OpenCV.

This module keeps the responsibilities narrow:
- open a video file
- expose reliable metadata (fps, frame count, size)
- read individual frames by index or sequentially
- cleanly release resources / support context manager usage

It is intentionally UI-agnostic so it can be reused by a future ring buffer,
Qt viewer, click-mapping overlays, or data logging components.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional, Tuple

import numpy as np

try:
    import cv2
except ImportError as exc:  # pragma: no cover - defensive import guard
    raise ImportError(
        "video_io.py requires OpenCV. Install with: pip install opencv-python"
    ) from exc


@dataclass(frozen=True)
class VideoMetadata:
    """Simple container for frequently-used video properties."""

    path: Path
    frame_count: int
    fps: float
    width: int
    height: int
    duration: float  # seconds

    @property
    def size(self) -> Tuple[int, int]:
        """Return (width, height)."""

        return self.width, self.height


class VideoReader:
    """Small wrapper around cv2.VideoCapture for frame-accurate access.

    Designed to stay lightweight while offering conveniences that higher-level
    components (ring buffer, UI) can depend on.
    """

    def __init__(self, path: Path | str, convert_to_rgb: bool = True) -> None:
        self.path = Path(path).expanduser().resolve()
        self.convert_to_rgb = convert_to_rgb
        self._cap: Optional[cv2.VideoCapture] = None
        self._meta: Optional[VideoMetadata] = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def open(self) -> "VideoReader":
        """Open the video file and populate metadata."""

        if self._cap is not None:
            return self

        cap = cv2.VideoCapture(str(self.path))
        if not cap.isOpened():
            raise FileNotFoundError(f"Could not open video: {self.path}")

        fps = float(cap.get(cv2.CAP_PROP_FPS)) or 0.0
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = frame_count / fps if fps > 0 else 0.0

        self._cap = cap
        self._meta = VideoMetadata(
            path=self.path,
            frame_count=frame_count,
            fps=fps,
            width=width,
            height=height,
            duration=duration,
        )
        return self

    def close(self) -> None:
        """Release the underlying VideoCapture."""

        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def __enter__(self) -> "VideoReader":
        return self.open()

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Metadata & state
    # ------------------------------------------------------------------
    @property
    def meta(self) -> VideoMetadata:
        if self._meta is None:
            raise RuntimeError("Video not opened. Call open() first.")
        return self._meta

    @property
    def current_index(self) -> int:
        """Return index of the next frame to be read (0-based)."""

        if self._cap is None:
            raise RuntimeError("Video not opened. Call open() first.")
        return int(self._cap.get(cv2.CAP_PROP_POS_FRAMES))

    # ------------------------------------------------------------------
    # Reading
    # ------------------------------------------------------------------
    def set_position(self, frame_idx: int) -> int:
        """Seek to ``frame_idx`` (0-based). Returns the position set."""

        if self._cap is None:
            raise RuntimeError("Video not opened. Call open() first.")
        frame_idx = max(0, int(frame_idx))
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        return self.current_index

    def read(self, frame_idx: Optional[int] = None) -> Tuple[int, Optional[np.ndarray]]:
        """
        Read a frame.

        Args:
            frame_idx: Optional absolute frame index to jump to before reading.

        Returns:
            (index, frame) where ``index`` is the 0-based frame number of the
            returned frame. ``frame`` is a numpy array (H, W, 3). If EOF is hit,
            returns (-1, None).
        """

        if self._cap is None:
            raise RuntimeError("Video not opened. Call open() first.")

        if frame_idx is not None:
            self.set_position(frame_idx)

        ok, frame = self._cap.read()
        if not ok or frame is None:
            return -1, None

        idx = self.current_index - 1  # read() advances the internal pointer

        if self.convert_to_rgb:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        return idx, frame

    def iter_frames(
        self, start: int = 0, step: int = 1, stop: Optional[int] = None
    ) -> Iterator[Tuple[int, np.ndarray]]:
        """Generator yielding (index, frame) pairs.

        Args:
            start: First frame index to read.
            step: Frame stride (e.g., 1 for every frame, 10 to sample sparsely).
            stop: Optional exclusive upper bound on frame index.
        """

        if self._cap is None:
            raise RuntimeError("Video not opened. Call open() first.")

        stop = self.meta.frame_count if stop is None else min(stop, self.meta.frame_count)
        idx = self.set_position(start)
        while idx < stop:
            idx, frame = self.read()
            if frame is None or idx < 0:
                break
            yield idx, frame
            idx = self.set_position(idx + step)


def probe_video(path: Path | str) -> VideoMetadata:
    """Convenience function: open a video, fetch metadata, and close it."""

    with VideoReader(path) as vr:
        return vr.meta


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Quick video probe & sampler")
    parser.add_argument("video", type=str, help="Path to the video file")
    parser.add_argument(
        "--sample",
        type=int,
        default=5,
        help="Number of frames to sample sequentially (default: 5)",
    )
    parser.add_argument(
        "--start",
        type=int,
        default=0,
        help="Starting frame index for sampling",
    )
    parser.add_argument(
        "--step",
        type=int,
        default=1,
        help="Frame stride between samples",
    )

    args = parser.parse_args()

    vr = VideoReader(args.video)
    with vr as reader:
        meta = reader.meta
        print(
            f"Opened {meta.path}\n"
            f"  Frames: {meta.frame_count}\n"
            f"  FPS:    {meta.fps:.3f}\n"
            f"  Size:   {meta.width}x{meta.height}\n"
            f"  Duration: {meta.duration:.2f} s"
        )

        print("\nSampling frames:")
        count = 0
        for idx, frame in reader.iter_frames(start=args.start, step=args.step):
            print(f"  frame {idx}: shape={frame.shape}")
            count += 1
            if count >= args.sample:
                break

