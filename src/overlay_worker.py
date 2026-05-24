"""
Simple overlay worker: consumes UI events, produces overlay instructions.

Contracts
---------
Incoming events (dict):
  - {'type': 'click', 'frame': int, 'x': float, 'y': float, 'button': 'left'/'right', 'phase': 'down'/'up', 't': float}
  - {'type': 'seek', 'target': int, 't': float}
  - {'type': 'play_state', 'playing': bool, 'frame': Optional[int], 't': float}

Outgoing render instruction:
  {'frame': int, 'overlays': [overlay-spec, ...], 'status': str, 't': float}

Overlay spec examples:
  {'kind': 'dot', 'x': float, 'y': float, 'r': int, 'color': '#ff0000'}
  {'kind': 'line', 'points': [(x1,y1), (x2,y2)], 'color': '#00ff00', 'width': 2}

This worker is intentionally minimal: it draws a path of click positions
(using "down" events) per frame and sends an overlay for that frame.
"""

from __future__ import annotations

import threading
import time
from queue import Queue, Empty
from typing import Dict, List, Tuple, Optional, Any


class OverlayWorker(threading.Thread):
    def __init__(self, event_queue: Queue, render_queue: Queue, video_info: Optional[Dict[str, Any]] = None):
        super().__init__(name="overlay-worker", daemon=True)
        self.events = event_queue
        self.render = render_queue
        self._stop = threading.Event()
        self._frame_points: Dict[str, Dict[int, Tuple[float, float]]] = {}  # track -> frame -> (x, y)
        self._frame_overlays: Dict[int, List[dict]] = {}  # persistent per-frame overlays
        # Session metadata + raw point history (video-space coordinates)
        self.video_info: Dict[str, Any] = video_info or {}
        self.point_history: List[Dict[str, Any]] = []  # each: {frame, x, y, t, button}
        self._listeners: List[Any] = []  # callbacks to notify when point_history updates
        self.track_colors: Dict[str, str] = {}
        self.marker_size: int = 5
        self.marker_alpha: float = 1.0
        self.onion_frames: int = 30  # default onionskin duration (frames forward/back)

    def stop(self) -> None:
        self._stop.set()

    def run(self) -> None:
        while not self._stop.is_set():
            try:
                evt = self.events.get(timeout=0.1)
            except Empty:
                continue

            if not isinstance(evt, dict) or "type" not in evt:
                continue

            #+++++++++++++++++++++++++++++++++++#
            #Check event conditions and create overlays accordingly below with if statements#
            #+++++++++++++++++++++++++++++++++++#

            etype = evt["type"]
            now = time.time()

            if etype == "seek":
                # keep stored points; no overlays cleared on seek
                continue

            if etype == "track_meta":
                tid = str(evt.get("track_id", "0"))
                color = evt.get("color")
                if color:
                    self.track_colors[tid] = color
                continue

            if etype == "options":
                sz = evt.get("marker_size")
                al = evt.get("marker_alpha")
                onion = evt.get("onion_frames")
                changed = False
                if sz is not None:
                    try:
                        self.marker_size = max(1, int(sz))
                        changed = True
                    except Exception:
                        pass
                if al is not None:
                    try:
                        self.marker_alpha = max(0.05, min(1.0, float(al)))
                        changed = True
                    except Exception:
                        pass
                if onion is not None:
                    try:
                        self.onion_frames = max(0, min(1000, int(onion)))
                        changed = True
                    except Exception:
                        pass
                if changed:
                    frames_payload = self._build_overlays_with_onion()
                    msg = {"clear_all": True, "t": now}
                    if frames_payload:
                        msg["frames"] = frames_payload
                    self._enqueue_render(msg)
                continue

            if etype == "click":
                frame_idx = evt.get("frame_idx", evt.get("frame"))
                track_id = str(evt.get("track_id", "0"))
                if evt.get("phase") == "down":
                    # Use base-frame coordinates (pre view-transform) so overlays align with zoom/pan
                    x = evt.get("x_base")
                    y = evt.get("y_base")
                    if x is None or y is None:
                        # graceful fallback for older events
                        x = evt.get("x_disp", evt.get("x"))
                        y = evt.get("y_disp", evt.get("y"))
                    if x is None or y is None:
                        continue
                    # replace point for this frame
                    self._frame_points.setdefault(track_id, {})[frame_idx] = (x, y)
                    # Persist raw (video-space) point data for downstream use
                    self.point_history.append(
                        {
                            "frame": frame_idx,
                            "x": evt.get("x"),  # original video coordinates
                            "y": evt.get("y"),
                            "track_id": track_id,
                            "x_cal": evt.get("x_cm", evt.get("x")),
                            "y_cal": evt.get("y_cm", evt.get("y")),
                            "dpcm": evt.get("dpcm", 1.0),
                            "t": evt.get("t"),
                            "button": evt.get("button"),
                        }
                    )
                    self._notify_point_listeners()
                    # rebuild overlays with onionskin across frames
                    frames_payload = self._build_overlays_with_onion()
                    if frames_payload:
                        msg = {"frames": frames_payload, "t": now}
                        self._enqueue_render(msg)
                continue

            if etype == "play_state":
                # no overlay, just state tracking if desired
                continue

    def _build_base_overlays(self) -> Dict[int, List[dict]]:
        """Build per-frame overlay lists from current point registry."""
        overlays = {}
        if not self._frame_points:
            return overlays
        for track_id, frames_map in self._frame_points.items():
            sorted_items = sorted(frames_map.items())
            path_so_far: List[Tuple[float, float]] = []
            color = self.track_colors.get(track_id, "#4caf50")
            r = max(1, int(self.marker_size))
            width = max(1, int(round(self.marker_size * 0.6)))
            base_alpha = float(self.marker_alpha)
            for fid, (x, y) in sorted_items:
                path_so_far.append((x, y))
                items = overlays.setdefault(fid, [])
                items.append({"kind": "dot", "x": x, "y": y, "r": r, "color": color, "alpha": base_alpha})
                if len(path_so_far) >= 2:
                    items.append(
                        {
                            "kind": "line",
                            "points": path_so_far[-2:].copy(),
                            "color": color,
                            "width": width,
                            "alpha": base_alpha,
                        }
                    )
        return overlays

    def _fade_series(self, start_alpha: float, min_alpha: float) -> List[float]:
        """Generate a monotonic fade curve across onion frames."""
        if self.onion_frames <= 0:
            return []
        if self.onion_frames == 1:
            return [max(min_alpha, start_alpha)]
        # choose decay so the last onion frame lands near min_alpha
        decay = (min_alpha / start_alpha) ** (1.0 / (self.onion_frames - 1))
        return [max(min_alpha, start_alpha * (decay ** offset)) for offset in range(self.onion_frames)]

    def _build_overlays_with_onion(self) -> List[dict]:
        """Return a frames payload with base overlays plus onionskin (prev frames faded)."""
        base = self._build_base_overlays()
        if not base:
            return []
        point_frames = sorted(base.keys())
        overlays_map: Dict[int, List[dict]] = {fid: list(ovs) for fid, ovs in base.items()}

        # backward onionskin (into current frame)
        back_fades = self._fade_series(start_alpha=0.4, min_alpha=0.05)
        for idx, fid in enumerate(point_frames):
            for offset, fade in enumerate(back_fades, start=1):
                prev_idx = idx - offset
                if prev_idx >= 0:
                    prev_frame = point_frames[prev_idx]
                    for ov in base[prev_frame]:
                        overlays_map[fid].append({**ov, "alpha": ov.get("alpha", 1.0) * fade})

        # forward onionskin: project each frame's overlays onto future frames
        fwd_fades = self._fade_series(start_alpha=0.6, min_alpha=0.1)
        for fid in point_frames:
            for fade_idx, fade in enumerate(fwd_fades, start=1):
                target = fid + fade_idx
                for ov in base[fid]:
                    overlays_map.setdefault(target, []).append({**ov, "alpha": ov.get("alpha", 1.0) * fade})

        # build payload with clear=True to fully refresh each frame's overlays
        payload = []
        for fid, ovs in overlays_map.items():
            payload.append({"frame": fid, "overlays": ovs, "ttl": None, "clear": True})
        return payload
            #+++++++++++++++++++++++++++++++++++#

    def register_point_listener(self, cb: Any) -> None:
        """Register a callback to be invoked when point_history changes."""
        if cb is None:
            return
        self._listeners.append(cb)

    def _notify_point_listeners(self) -> None:
        for cb in list(self._listeners):
            try:
                cb(self.point_history, self.video_info)
            except Exception:
                # keep going even if one listener fails
                continue
    def _enqueue_render(self, msg: dict) -> None:
        try:
            self.render.put_nowait(msg)
        except Exception:
            # drop if render queue is full
            pass
