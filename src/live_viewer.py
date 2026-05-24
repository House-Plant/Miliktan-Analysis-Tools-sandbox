"""
Tiny Tkinter live viewer that reads frames via VideoReader + FrameRingBuffer.

Purpose: demonstrate end-to-end flow before building a richer Qt UI.
 - spawns a decoder thread that fills a ring buffer
 - displays the latest available frame in a Tk window
 - stops automatically when decoding finishes
"""

from __future__ import annotations

import argparse
import threading
import tkinter as tk
from tkinter import filedialog, ttk
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple, List, Dict
from queue import Queue, Empty
import time
import csv

import numpy as np
from PIL import Image, ImageTk, ImageDraw, ImageColor
try:
    import cv2  # for fast interaction resizes
except ImportError:  # pragma: no cover
    cv2 = None

from buffer import FrameRingBuffer, start_decoder_controlled
from video_io import probe_video
from overlay_worker import OverlayWorker


def _resize_frame(frame: np.ndarray, target_width: Optional[int]) -> np.ndarray:
    """Optionally resize keeping aspect ratio; returns new array."""

    if target_width is None:
        return frame
    h, w = frame.shape[:2]
    if w == 0 or h == 0:
        return frame
    scale = target_width / float(w)
    new_size = (target_width, int(h * scale))
    img = Image.fromarray(frame)
    # Use nearest for integer scaling to preserve pixel shapes; otherwise lanczos.
    resample = Image.Resampling.NEAREST if abs(scale - round(scale)) < 1e-6 else Image.Resampling.LANCZOS
    img = img.resize(new_size, resample)
    return np.asarray(img)


def _parse_color(color: str, alpha: float = 1.0) -> tuple:
    rgba = ImageColor.getrgb(color)
    a = max(0, min(255, int(alpha * 255)))
    if len(rgba) == 4:
        return (*rgba[:3], a)
    return (*rgba, a)


def apply_overlays(frame: np.ndarray, overlays: list) -> np.ndarray:
    """Draw overlay primitives onto a copy of the frame (supports alpha)."""

    if not overlays:
        return frame
    base = Image.fromarray(frame).convert("RGBA")
    overlay_img = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay_img)

    for ov in overlays:
        kind = ov.get("kind")
        alpha = float(ov.get("alpha", 1.0))
        color = ov.get("color", "#ff0000")
        rgba = _parse_color(color, alpha)
        if kind == "dot":
            x, y = float(ov["x"]), float(ov["y"])
            r = int(ov.get("r", 4))
            draw.ellipse((x - r, y - r, x + r, y + r), fill=rgba, outline=rgba)
        elif kind == "line":
            pts = ov.get("points") or []
            if len(pts) >= 2:
                width = int(ov.get("width", 2))
                draw.line(pts, fill=rgba, width=width)
    composed = Image.alpha_composite(base, overlay_img)
    return np.asarray(composed.convert("RGB"))



class LiveViewer:
    def __init__(
        self,
        video_path: Path,
        capacity: int = 120,
        step: int = 1,
        refresh_ms: int = 33,
        width: Optional[int] = None,
    ) -> None:
        self.video_path = Path(video_path)
        self.refresh_ms = max(1, int(refresh_ms))
        self.target_width = width
        self.meta = probe_video(self.video_path)
        self._playing = True
        self._pending_step = False
        self.display_scale = 1.0

        self.buffer = FrameRingBuffer(capacity=capacity)
        self.command_queue, self.decoder_thread = start_decoder_controlled(
            self.video_path, self.buffer, start=0, step=step, block_on_full=True
        )

        # Event/render queues for out-of-UI processing
        # Use unbounded queues to avoid losing control/option events under bursty input
        self.event_queue: Queue = Queue(maxsize=0)
        self.render_queue: Queue = Queue(maxsize=0)
        self.overlay_cache: Dict[int, Dict[str, object]] = {}  # frame -> {"overlays": [...], "expires": ts}
        video_info = {
            "path": str(self.video_path.resolve()),
            "name": self.video_path.name,
            "width": self.meta.width,
            "height": self.meta.height,
            "duration": self.meta.duration,
            "frame_count": self.meta.frame_count,
            "fps": self.meta.fps,
        }
        # Tracking (multiple paths) state
        self.tracks: List[Dict[str, object]] = []
        self.active_track_id: str = "track1"
        self._track_colors = [
            "#ff5722",
            "#4caf50",
            "#2196f3",
            "#9c27b0",
            "#ffc107",
            "#009688",
            "#e91e63",
            "#3f51b5",
        ]
        self._track_var: Optional[tk.StringVar] = None
        self.point_history: List[dict] = []
        self.video_info: Dict[str, object] = video_info
        self.root = tk.Tk()
        screen_w = self.root.winfo_screenwidth()
        screen_h = self.root.winfo_screenheight()

        self._init_tracks()
        self._worker = OverlayWorker(self.event_queue, self.render_queue, video_info=video_info)
        self._worker.register_point_listener(self._on_points_updated)
        self._worker.start()

        # Compute the largest pixel-perfect integer scale that fits; downscale if needed
        max_w = max(1, int(screen_w * 0.9))
        max_h = max(1, int(screen_h * 0.9) - 120)  # leave room for status bar/buttons
        if self.meta.width and self.meta.height:
            fit_scale = min(max_w / self.meta.width, max_h / self.meta.height)
        else:
            fit_scale = 1.0

        # Prefer integer upscales for crisp pixels; allow fractional downscale to fit
        if fit_scale >= 1.0:
            base_scale = max(1.0, float(int(fit_scale)))
        else:
            base_scale = max(0.1, fit_scale)

        if self.target_width is not None and self.meta.width:
            requested_scale = self.target_width / float(self.meta.width)
            self.display_scale = max(0.1, min(requested_scale, fit_scale))
        else:
            self.display_scale = base_scale

        if self.meta.width:
            self.target_width = max(1, int(round(self.meta.width * self.display_scale)))
        else:
            self.target_width = None

        target_height = int(round(self.meta.height * self.display_scale)) if self.meta.height else 0

        # Clamp again in case rounding nudges us over the available area
        if self.meta.width and self.meta.height:
            if self.target_width > max_w or target_height > max_h:
                self.display_scale = min(max_w / self.meta.width, max_h / self.meta.height)
                self.target_width = max(1, int(round(self.meta.width * self.display_scale)))
                target_height = int(round(self.meta.height * self.display_scale))

        # Size window to video aspect ratio, bounded by screen
        aspect = self.meta.height / self.meta.width if self.meta.width else 1.0
        win_w = min((self.target_width or 0) + 40, int(screen_w * 0.95))
        win_h = min(target_height + 120, int(screen_h * 0.95))
        self.root.geometry(f"{win_w}x{win_h}+20+20")
        self.root.title(f"Live Viewer - {self.video_path.name}")
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.root.bind("<KeyPress-q>", lambda evt: self._on_close())
        self.root.bind("<Escape>", lambda evt: self._on_close())
        self.root.bind("<space>", lambda evt: self._toggle_play())
        self.root.bind("<Left>", lambda evt: self._seek_relative(-1))
        self.root.bind("<Right>", lambda evt: self._seek_relative(1))
        self.root.bind("<Shift-Left>", lambda evt: self._seek_relative(-10))
        self.root.bind("<Shift-Right>", lambda evt: self._seek_relative(10))
        self.root.bind("<period>", lambda evt: self._seek_relative(30))   # fast forward
        self.root.bind("<comma>", lambda evt: self._seek_relative(-30))   # fast reverse
        self.root.bind("<Control-s>", lambda evt: self._open_save_dialog())
        self.root.bind("<Control-m>", lambda evt: self._enter_measure_mode())
        self.root.bind("<BackSpace>", lambda evt: self._cancel_measure_mode())
        self.root.bind("<Return>", lambda evt: self._confirm_measure_point())
        self.root.bind("<KeyRelease-Shift_L>", lambda evt: self._clear_pan_anchor())
        self.root.bind("<KeyRelease-Shift_R>", lambda evt: self._clear_pan_anchor())

        self.label = tk.Label(self.root)
        self.label.pack(fill="both", expand=True)
        self.label.bind("<ButtonPress-1>", lambda evt: self._handle_click(evt, "down"))
        self.label.bind("<ButtonRelease-1>", lambda evt: self._handle_click(evt, "up"))
        self.label.bind("<ButtonPress-3>", lambda evt: self._handle_click(evt, "down"))
        self.label.bind("<ButtonRelease-3>", lambda evt: self._handle_click(evt, "up"))
        # Zoom with wheel
        self.label.bind("<MouseWheel>", self._on_zoom)  # Windows / macOS
        self.label.bind("<Button-4>", lambda e: self._on_zoom(e, delta=120))  # X11 scroll up
        self.label.bind("<Button-5>", lambda e: self._on_zoom(e, delta=-120))  # X11 scroll down
        # Pan with Shift + mouse move
        self.label.bind("<Motion>", self._on_motion_pan)
        # Tiny viewport minimap in top-right
        self._minimap_size = (110, 80)
        self._minimap_margin = 2
        self._minimap_canvas = tk.Canvas(
            self.label,
            width=self._minimap_size[0],
            height=self._minimap_size[1],
            highlightthickness=0,
            bd=0,
            bg=self.root.cget("background"),
        )
        self._minimap_canvas.place(relx=1.0, y=6, anchor="ne")
        # Anchor marker popup
        self._anchor_popup: Optional[tk.Toplevel] = None

        # Top-center dropdown tab overlaid on the video label
        self._dropdown_open = False
        self._dropdown_animating = False
        self._dropdown_panel: Optional[tk.Frame] = None
        self._dropdown_height = 0
        self._dropdown_target_height = 90  # initial fallback; will auto-size on open
        self._dropdown_hide_popup: Optional[tk.Toplevel] = None
        self._controls_row: Optional[tk.Frame] = None  # single row for all dropdown buttons
        self._init_styles()
        self._init_dropdown_menu()

        self.status = tk.Label(self.root, text="Starting...", anchor="w")
        self.status.pack(fill="x")
        # Thin progress bar along the bottom to show playback position
        self._progress_max = max(1, self.meta.frame_count - 1)
        self.progress = ttk.Progressbar(
            self.root,
            orient="horizontal",
            mode="determinate",
            maximum=self._progress_max,
            style="Thin.Horizontal.TProgressbar",
        )
        self.progress.pack(fill="x", side="bottom")
        self.root.focus_force()

        self._photo = None  # keep reference to avoid GC
        self._last_idx = None
        self._last_frame_shape: Optional[Tuple[int, int]] = None  # (h, w) original
        self._last_display_shape: Optional[Tuple[int, int]] = None  # (h, w) resized
        self._last_frame_base: Optional[np.ndarray] = None  # resized frame without overlays
        self.click_events = []
        self._save_dialog: Optional[tk.Toplevel] = None
        self._save_name_var: Optional[tk.StringVar] = None
        self._save_path_var: Optional[tk.StringVar] = None
        self._save_name_entry: Optional[tk.Entry] = None
        self._save_path_entry: Optional[tk.Entry] = None
        self._save_browse_btn: Optional[tk.Button] = None
        self._last_save_target: Optional[Path] = None
        # Use lab2/cache directory (sibling to src)
        self._cache_dir = Path(__file__).resolve().parent.parent / "cache"
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._save_path_cache_file = self._cache_dir / "save_path.txt"
        self._measure_mode = False
        self._measure_hint = (
            "Adding a measure: Please click two points, and then enter their physical measure (backspace to abort)"
        )
        self._measure_clicks: List[dict] = []
        self._measure_overlay_cache: Optional[Image.Image] = None
        self._measure_overlay_sig: Optional[Tuple[int, int, str]] = None
        self._measure_prompt: str = "Please select first point (enter to confirm)"
        self._measure_enter_blocked: bool = True
        self._measure_phase: str = "first"  # "first", "second", "length"
        # measure points stored in base-frame coords; originals kept for accurate distances
        self._measure_p1: Optional[Tuple[float, float]] = None  # base-frame coords
        self._measure_p2: Optional[Tuple[float, float]] = None  # base-frame coords
        self._measure_p1_orig: Optional[Tuple[float, float]] = None  # original video coords
        self._measure_p2_orig: Optional[Tuple[float, float]] = None
        self._measure_popup: Optional[tk.Toplevel] = None
        self._measure_mant_var: Optional[tk.StringVar] = None
        self._measure_unit_var: Optional[tk.StringVar] = None
        self._measure_dpi_var: Optional[tk.StringVar] = None
        self._measure_error_var: Optional[tk.StringVar] = None
        self._measure_updating: bool = False
        self._measure_equation_var: Optional[tk.StringVar] = None
        self._measure_result: Optional[dict] = None  # stores last confirmed measurement
        self._current_dpcm: float = 1.0
        # Marker/options state
        self._marker_size: int = 5
        self._marker_alpha: float = 1.0
        self._marker_size_var: Optional[tk.IntVar] = None
        self._marker_alpha_var: Optional[tk.DoubleVar] = None
        self._options_win: Optional[tk.Toplevel] = None
        self._show_grid: bool = False
        self._grid_var: Optional[tk.BooleanVar] = None
        # Zoom tuning (higher = faster)
        self._zoom_speed_gain: float = 300.0
        self._zoom_speed_var: Optional[tk.DoubleVar] = None
        self._pan_speed_gain: float = 0.6  # lower = slower pan
        self._pan_speed_var: Optional[tk.DoubleVar] = None
        self._onion_frames: int = 30
        self._onion_slider_var: Optional[tk.IntVar] = None  # 0-100 slider value mapped logarithmically
        self._hide_markers: bool = False
        self._hide_ui: bool = False
        self._hide_markers_btn: Optional[tk.Button] = None
        self._hide_ui_btn: Optional[tk.Button] = None
        self._ui_restore_binding: Optional[str] = None
        # Zoom interaction smoothing
        self._last_zoom_ts: float = 0.0
        self._zoom_fast_window: float = 0.12  # seconds to prefer faster resample after scroll
        self._zoom_seq: int = 0  # increments each zoom gesture
        self._pending_zoom_redraw: bool = False
        self._zoom_redraw_scheduled: bool = False
        # View transform
        self._zoom_scale: float = 1.0
        self._pan_x: float = 0.0  # pixels in frame_base coords relative to center
        self._pan_y: float = 0.0
        self._view_x0 = 0.0
        self._view_y0 = 0.0
        self._view_w = None
        self._view_h = None
        self._suppress_click_once: bool = False  # used to ignore first click after restoring UI
        self._pan_anchor: Optional[Tuple[float, float]] = None  # cursor pos (display space) when shift engaged
        self._pan_cursor: Optional[Tuple[float, float]] = None  # current cursor in display space while panning
        # display->frame mapping (set each redraw)
        self._disp_sx = 1.0
        self._disp_sy = 1.0
        self._disp_ox = 0.0
        self._disp_oy = 0.0
        self._auto_fit_done: bool = False
        self._auto_fit_attempts: int = 0
        # push initial options so worker renders with current settings
        self._publish_event(
            {
                "type": "options",
                "marker_size": self._marker_size,
                "marker_alpha": self._marker_alpha,
                "onion_frames": self._onion_frames,
            }
        )
        # Prebuild dropdown controls so dark styling applies immediately on first open
        self._ensure_dropdown_panel()

    # ----------------------- lifecycle ----------------------- #
    def start(self) -> None:
        self._tick()
        self.root.mainloop()

    def _on_close(self) -> None:
        self._send_command({"type": "stop"})
        self.buffer.stop()
        if self.decoder_thread and self.decoder_thread.is_alive():
            self.decoder_thread.join(timeout=1.0)
        if self._worker and self._worker.is_alive():
            self._worker.stop()
            self._worker.join(timeout=1.0)
        self._destroy_options_panel()
        self.root.destroy()

    def _send_command(self, cmd: dict) -> None:
        try:
            self.command_queue.put_nowait(cmd)
        except Exception:
            pass

    def _status_text(self, base: str) -> str:
        """Status line formatter (kept simple; hint now overlaid on video)."""
        return base

    def _on_points_updated(self, history: list, video_info: dict) -> None:
        """Receive point history + video metadata from overlay worker."""
        self.point_history = list(history) if history is not None else []
        if video_info:
            self.video_info = dict(video_info)

    def _apply_measure_overlay(self, frame: np.ndarray) -> np.ndarray:
        """Overlay a subtle dim and guidance text onto the frame."""
        if frame is None:
            return frame
        h, w = frame.shape[:2]
        sig = (w, h, self._measure_prompt)
        if self._measure_overlay_cache is None or self._measure_overlay_sig != sig:
            base_overlay = Image.new("RGBA", (w, h), (0, 0, 0, 60))  # ~24% dim
            draw = ImageDraw.Draw(base_overlay)
            lines = [self._measure_prompt, self._measure_hint] if self._measure_prompt else [self._measure_hint]
            margin = 12
            y_cursor = h - margin
            for text in reversed([ln for ln in lines if ln]):
                try:
                    bbox = draw.textbbox((0, 0), text)
                    text_w = bbox[2] - bbox[0]
                    text_h = bbox[3] - bbox[1]
                except Exception:
                    text_w, text_h = draw.textsize(text)
                y_cursor -= text_h
                x = max(margin, (w - text_w) // 2)
                pad = 6
                rect = (x - pad, y_cursor - pad, x + text_w + pad, y_cursor + text_h + pad)
                draw.rectangle(rect, fill=(0, 0, 0, 140))
                draw.text((x, y_cursor), text, fill=(255, 255, 255, 255))
                y_cursor -= 4  # spacing
            self._measure_overlay_cache = base_overlay
            self._measure_overlay_sig = sig

        overlay = self._measure_overlay_cache.copy()
        draw = ImageDraw.Draw(overlay)
        # Markers and line
        r = 6
        def base_to_disp(pt: Tuple[float, float]) -> Optional[Tuple[float, float]]:
            if pt is None or self._view_w is None or self._view_h is None:
                return None
            try:
                x_disp = (pt[0] - (self._view_x0 or 0.0)) * (w / self._view_w)
                y_disp = (pt[1] - (self._view_y0 or 0.0)) * (h / self._view_h)
                return (x_disp, y_disp)
            except Exception:
                return None

        p1 = base_to_disp(self._measure_p1)
        p2 = base_to_disp(self._measure_p2)
        if p1 is not None:
            x1, y1 = p1
            draw.line((x1 - r, y1, x1 + r, y1), fill=(0, 200, 0, 255), width=2)
            draw.line((x1, y1 - r, x1, y1 + r), fill=(0, 200, 0, 255), width=2)
        if p2 is not None:
            x2, y2 = p2
            draw.line((x2 - r, y2, x2 + r, y2), fill=(0, 200, 0, 255), width=2)
            draw.line((x2, y2 - r, x2, y2 + r), fill=(0, 200, 0, 255), width=2)
        if p1 is not None and p2 is not None:
            draw.line((p1[0], p1[1], p2[0], p2[1]), fill=(0, 200, 0, 255), width=2)

        base = Image.fromarray(frame).convert("RGBA")
        composed = Image.alpha_composite(base, overlay)
        return np.asarray(composed.convert("RGB"))

    def _refresh_measure_frame(self) -> None:
        """Redraw current frame with measure overlay (if available)."""
        if not self._measure_mode:
            return
        self._redraw_current_frame(apply_measure=True)
        self._update_minimap()

    def _redraw_current_frame(self, apply_measure: bool = False) -> None:
        """Redraw using last frame + overlays + view (used after measure/zoom/pan changes)."""
        if self._last_frame_base is None:
            return
        overlays_payload = self.overlay_cache.get(self._last_idx) if self._last_idx is not None else None
        overlays = overlays_payload["overlays"] if overlays_payload else []
        overlays = self._visible_overlays(overlays)
        frame = apply_overlays(self._last_frame_base.copy(), overlays) if overlays else self._last_frame_base
        frame = self._apply_view_transform(frame)
        frame = self._apply_grid(frame)
        if apply_measure and self._measure_mode:
            frame = self._apply_measure_overlay(frame)
        img = Image.fromarray(frame)
        self._photo = ImageTk.PhotoImage(image=img)
        self.label.configure(image=self._photo)

    def _enter_measure_mode(self) -> None:
        """Activate measure mode: dim window and show guidance text."""
        if self._measure_mode:
            return
        self._measure_mode = True
        self._measure_overlay_cache = None
        self._measure_overlay_sig = None
        self._measure_prompt = "Please select first point (enter to confirm)"
        self._measure_enter_blocked = True
        self._measure_phase = "first"
        self._measure_p1 = None
        self._measure_p2 = None
        self._measure_p1_orig = None
        self._measure_p2_orig = None
        self._measure_popup = None
        # Force status refresh
        if self._last_idx is not None:
            mode = "playing" if self._playing else "paused"
            self.status.configure(
                text=self._status_text(f"Frame {self._last_idx} | {mode} | buffer={len(self.buffer)}/{self.buffer.capacity}")
            )
        else:
            self.status.configure(text=self._status_text("Adding a measure..."))
        self._measure_clicks.clear()
        # Force immediate overlay application on current displayed frame
        if self._last_frame_base is not None:
            overlays_payload = self.overlay_cache.get(self._last_idx) if self._last_idx is not None else None
            overlays = overlays_payload["overlays"] if overlays_payload else []
            overlays = self._visible_overlays(overlays)
            frame = apply_overlays(self._last_frame_base.copy(), overlays) if overlays else self._last_frame_base
            frame = self._apply_measure_overlay(frame)
            img = Image.fromarray(frame)
            self._photo = ImageTk.PhotoImage(image=img)
            self.label.configure(image=self._photo)

    def _cancel_measure_mode(self) -> None:
        """Exit measure mode and remove overlay hint."""
        if not self._measure_mode:
            return
        self._measure_mode = False
        self._measure_overlay_cache = None
        self._measure_overlay_sig = None
        self._measure_prompt = ""
        self._measure_enter_blocked = True
        self._measure_phase = "first"
        self._measure_p1 = None
        self._measure_p2 = None
        self._measure_p1_orig = None
        self._measure_p2_orig = None
        self._close_measure_popup()
        # Refresh status without hint
        if self._last_idx is not None:
            mode = "playing" if self._playing else "paused"
            self.status.configure(
                text=self._status_text(f"Frame {self._last_idx} | {mode} | buffer={len(self.buffer)}/{self.buffer.capacity}")
            )
        else:
            self.status.configure(text=self._status_text("Ready"))
        self._measure_clicks.clear()
        # Force immediate redraw without overlay
        if self._last_frame_base is not None and self._photo is not None:
            overlays_payload = self.overlay_cache.get(self._last_idx) if self._last_idx is not None else None
            overlays = overlays_payload["overlays"] if overlays_payload else []
            frame = apply_overlays(self._last_frame_base.copy(), overlays) if overlays else self._last_frame_base
            img = Image.fromarray(frame)
            self._photo = ImageTk.PhotoImage(image=img)
            self.label.configure(image=self._photo)
        self._restore_focus_to_main()

    # ----------------------- dialogs ----------------------- #
    def _open_save_dialog(self) -> None:
        """Open (or focus) a placeholder save dialog invoked via Ctrl+S."""

        if self._save_dialog is not None and self._save_dialog.winfo_exists():
            self._save_dialog.lift()
            self._save_dialog.focus_force()
            if self._save_name_var is not None:
                if self._save_name_entry is not None and self._save_name_entry.winfo_exists():
                    self._save_name_entry.focus_set()
            return

        if self._save_name_var is None:
            default_name = f"{self.video_path.stem}_pointtrack.csv"
            self._save_name_var = tk.StringVar(value=default_name)
        if self._save_path_var is None:
            cached = self._load_cached_save_path()
            self._save_path_var = tk.StringVar(value=cached or str(self.video_path.parent))

        dlg = tk.Toplevel(self.root)
        self._save_dialog = dlg
        dlg.title("Save")
        dlg.transient(self.root)
        dlg.geometry("360x200+60+60")
        dlg.focus_force()
        dlg.protocol("WM_DELETE_WINDOW", self._close_save_dialog)

        # Simple filename entry UI (selectable with mouse, ready for typing)
        container = tk.Frame(dlg, background=self.root.cget("background"), padx=12, pady=12)
        container.pack(fill="both", expand=True)

        tk.Label(container, text="Save as:", anchor="w").pack(fill="x", pady=(0, 6))
        entry = tk.Entry(container, textvariable=self._save_name_var)
        entry.pack(fill="x")
        entry.focus_set()
        self._save_name_entry = entry

        # Path row with browse button
        path_row = tk.Frame(container, background=self.root.cget("background"))
        path_row.pack(fill="x", pady=(10, 0))
        path_entry = tk.Entry(path_row, textvariable=self._save_path_var)
        path_entry.pack(side="left", fill="x", expand=True)
        browse_btn = tk.Button(path_row, text="Browse...", command=self._browse_save_path)
        browse_btn.pack(side="left", padx=(8, 0))
        self._save_path_entry = path_entry
        self._save_browse_btn = browse_btn

        # Buttons row
        btn_row = tk.Frame(container, background=self.root.cget("background"))
        btn_row.pack(fill="x", pady=(12, 0))
        tk.Button(btn_row, text="Cancel", command=self._close_save_dialog, takefocus=0).pack(side="left")
        tk.Button(btn_row, text="Save", command=self._perform_save_dialog, takefocus=0).pack(side="right")

    def _close_save_dialog(self) -> None:
        if self._save_dialog is not None and self._save_dialog.winfo_exists():
            self._save_dialog.destroy()
        self._save_dialog = None
        self._save_name_entry = None
        self._save_path_entry = None
        self._save_browse_btn = None
        self._restore_focus_to_main()

    def _perform_save_dialog(self) -> None:
        """Write point history to CSV at chosen path."""
        try:
            save_dir = Path(self._save_path_var.get()).expanduser() if self._save_path_var else Path.cwd()
            save_name = self._save_name_var.get() if self._save_name_var else "pointtrack.csv"
            if not save_name.lower().endswith(".csv"):
                save_name += ".csv"
            target = (save_dir / save_name).resolve()
            save_dir.mkdir(parents=True, exist_ok=True)
            # cache chosen directory
            try:
                self._save_path_cache_file.write_text(str(save_dir))
            except Exception:
                pass

            # prepare track color lookup
            track_colors = {t["id"]: t.get("color") for t in self.tracks}

            # write csv
            headers = [
                "video_path",
                "video_name",
                "frame",
                "time_sec",
                "video_duration_sec",
                "track_id",
                "x_px",
                "y_px",
                "x_cal",
                "y_cal",
                "dpcm",
            ]
            with target.open("w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(headers)
                for row in self.point_history:
                    track_id = row.get("track_id", "0")
                    t_sec = None
                    if self.meta.fps and row.get("frame") is not None:
                        try:
                            t_sec = float(row.get("frame")) / float(self.meta.fps)
                        except Exception:
                            t_sec = None
                    writer.writerow(
                        [
                            self.video_info.get("path"),
                            self.video_info.get("name"),
                            row.get("frame"),
                            t_sec,
                            self.meta.duration,
                            track_id,
                            row.get("x"),
                            row.get("y"),
                            row.get("x_cal", row.get("x")),
                            row.get("y_cal", row.get("y")),
                            row.get("dpcm", 1.0),
                        ]
                    )
            self._last_save_target = target
        except Exception:
            self._last_save_target = None
        self._close_save_dialog()

    def _browse_save_path(self) -> None:
        """Open file dialog to select save path; cache selection."""
        initialdir = None
        if self._save_path_var is not None:
            try:
                maybe_dir = Path(self._save_path_var.get()).expanduser()
                if maybe_dir.is_dir():
                    initialdir = str(maybe_dir)
                elif maybe_dir.parent.is_dir():
                    initialdir = str(maybe_dir.parent)
            except Exception:
                pass

        filename = self._save_name_var.get() if self._save_name_var is not None else ""
        path = filedialog.asksaveasfilename(
            parent=self._save_dialog or self.root,
            initialdir=initialdir,
            initialfile=filename,
            title="Choose save location",
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
        )
        if path:
            if self._save_path_var is None:
                self._save_path_var = tk.StringVar()
            self._save_path_var.set(str(Path(path).parent))
            # cache chosen directory
            try:
                self._save_path_cache_file.write_text(self._save_path_var.get())
            except Exception:
                pass

    def _load_cached_save_path(self) -> Optional[str]:
        try:
            if self._save_path_cache_file.exists():
                txt = self._save_path_cache_file.read_text().strip()
                if txt:
                    return txt
        except Exception:
            return None
        return None

    def _toggle_play(self) -> None:
        self._playing = not self._playing
        if self._playing:
            self._send_command({"type": "pause", "value": False})
            self._pending_step = False
        else:
            self._send_command({"type": "pause", "value": True})
            self.buffer.clear()
            # realign decoder to current frame so resume doesn't jump ahead
            if self._last_idx is not None:
                self._send_command({"type": "seek", "frame": int(self._last_idx)})
        self._publish_event({"type": "play_state", "playing": self._playing, "frame": self._last_idx, "t": time.time()})

    def _seek_relative(self, delta: int) -> None:
        if self._last_idx is None:
            return
        target = max(0, min(self.meta.frame_count - 1, self._last_idx + delta))
        self._playing = False
        self._pending_step = True
        self.buffer.clear()
        self._send_command({"type": "step", "frame": target})
        self._publish_event({"type": "seek", "target": target, "t": time.time()})

    # ----------------------- UI loop ------------------------- #
    def _tick(self) -> None:
        now = time.time()
        # expire overlays
        for fid in list(self.overlay_cache.keys()):
            expires = self.overlay_cache[fid].get("expires")
            if expires and expires < now:
                self.overlay_cache.pop(fid, None)

        updated_current_overlay = False
        while True:
            try:
                msg = self.render_queue.get_nowait()
            except Empty:
                break
            if not isinstance(msg, dict):
                continue

            # optional clears
            if msg.get("clear_all"):
                self.overlay_cache.clear()
                updated_current_overlay = True
            if "clear_frames" in msg:
                for fid in msg.get("clear_frames") or []:
                    self.overlay_cache.pop(fid, None)

            frames_payload = msg.get("frames")
            if frames_payload:
                for entry in frames_payload:
                    frame_id = entry.get("frame")
                    if frame_id is None:
                        continue
                    overlays = entry.get("overlays", [])
                    alpha_override = entry.get("alpha")
                    if alpha_override is not None:
                        adjusted = []
                        for ov in overlays:
                            base_alpha = ov.get("alpha", 1.0)
                            adjusted.append({**ov, "alpha": base_alpha * alpha_override})
                        overlays = adjusted
                    ttl = entry.get("ttl")
                    expires = now + ttl if ttl else None
                    clear = bool(entry.get("clear", False))
                    existing = self.overlay_cache.get(frame_id, {"overlays": [], "expires": None})
                    if clear:
                        combined_overlays = overlays
                        combined_expires = expires
                    else:
                        combined_overlays = existing.get("overlays", []) + overlays
                        existing_exp = existing.get("expires")
                        if existing_exp is None or expires is None:
                            combined_expires = None
                        else:
                            combined_expires = max(existing_exp, expires)
                    self.overlay_cache[frame_id] = {"overlays": combined_overlays, "expires": combined_expires}
                    if self._last_idx is not None and frame_id == self._last_idx:
                        updated_current_overlay = True
            elif "frame" in msg:
                frame_id = msg["frame"]
                overlays = msg.get("overlays", [])
                ttl = msg.get("ttl")
                expires = now + ttl if ttl else None
                clear = bool(msg.get("clear", False))
                existing = self.overlay_cache.get(frame_id, {"overlays": [], "expires": None})
                if clear:
                    combined_overlays = overlays
                    combined_expires = expires
                else:
                    combined_overlays = existing.get("overlays", []) + overlays
                    existing_exp = existing.get("expires")
                    if existing_exp is None or expires is None:
                        combined_expires = None
                    else:
                        combined_expires = max(existing_exp, expires)
                self.overlay_cache[frame_id] = {"overlays": combined_overlays, "expires": combined_expires}
                if self._last_idx is not None and frame_id == self._last_idx:
                    updated_current_overlay = True

        pkt = None
        if self._playing:
            pkt = self.buffer.pop(block=False)
        elif self._pending_step:
            pkt = self.buffer.pop(block=False)
            if pkt is not None:
                self._pending_step = False

        if pkt is not None:
            orig_h, orig_w = pkt.frame.shape[:2]
            frame_base = _resize_frame(pkt.frame, self.target_width)
            self._last_frame_base = frame_base.copy()
            overlays_payload = self.overlay_cache.get(pkt.idx)
            overlays = overlays_payload["overlays"] if overlays_payload else []
            overlays = self._visible_overlays(overlays)
            frame = apply_overlays(frame_base, overlays) if overlays else frame_base
            frame = self._apply_view_transform(frame)
            frame = self._apply_grid(frame)
            if self._measure_mode:
                frame = self._apply_measure_overlay(frame)
            img = Image.fromarray(frame)
            self._photo = ImageTk.PhotoImage(image=img)
            self.label.configure(image=self._photo)
            if not self._auto_fit_done:
                self._fit_window_to_viewport()
            self._last_idx = pkt.idx
            self._last_frame_shape = (orig_h, orig_w)
            self._last_display_shape = frame.shape[:2]
            mode = "playing" if self._playing else "paused"
            self.status.configure(
                text=self._status_text(f"Frame {pkt.idx} | {mode} | buffer={len(self.buffer)}/{self.buffer.capacity}")
            )
        else:
            # keep showing last frame, just update status
            if updated_current_overlay and self._last_idx is not None and self._last_frame_base is not None:
                overlays_payload = self.overlay_cache.get(self._last_idx)
                overlays = overlays_payload["overlays"] if overlays_payload else []
                overlays = self._visible_overlays(overlays)
                frame = apply_overlays(self._last_frame_base.copy(), overlays) if overlays else self._last_frame_base
                frame = self._apply_view_transform(frame)
                frame = self._apply_grid(frame)
                if self._measure_mode:
                    frame = self._apply_measure_overlay(frame)
                img = Image.fromarray(frame)
                self._photo = ImageTk.PhotoImage(image=img)
                self.label.configure(image=self._photo)
                if not self._auto_fit_done:
                    self._fit_window_to_viewport()
            if self.buffer.stopped:
                state = "stopped"
            elif not self._playing:
                state = "paused"
            else:
                state = "waiting..."
            suffix = f" (last={self._last_idx})" if self._last_idx is not None else ""
            self.status.configure(text=self._status_text(f"{state}{suffix}"))

        # Continuous pan (if anchor is set) each tick; redraw if no new frame
        if self._pan_anchor is not None and self._pan_cursor is not None:
            self._apply_continuous_pan(redraw=(pkt is None))

        self._update_minimap()
        self._update_progress(self._last_idx)
        self.root.after(self.refresh_ms, self._tick)

    # ----------------------- Mouse handling ------------------ #
    def _handle_click(self, event: tk.Event, phase: str) -> None:
        """
        Capture mouse events on the image label.
        phase: 'down' or 'up'
        """
        if self._suppress_click_once:
            self._suppress_click_once = False
            return

        if self._last_idx is None or self._last_frame_shape is None or self._last_display_shape is None:
            return

        # Prefer actual photo size to avoid stale cached size (Tk returns these in the same coordinate
        # space as widget events, so no manual HiDPI scaling is needed)
        if self._photo is not None:
            disp_w = float(self._photo.width())
            disp_h = float(self._photo.height())
        else:
            disp_w = float(self._last_display_shape[1])
            disp_h = float(self._last_display_shape[0])

        # Tk event coordinates and widget sizes are already in the same logical pixel space as PhotoImage
        # even on HiDPI displays. The previous code divided by tk scaling, effectively double-dividing
        # on Retina and shrinking the clickable region to the top-left. Use raw values instead.
        label_w = max(1.0, float(self.label.winfo_width()))
        label_h = max(1.0, float(self.label.winfo_height()))
        orig_h, orig_w = self._last_frame_shape

        # account for image centering within the label (all in pixel units)
        offset_x = max(0.0, (label_w - disp_w) / 2.0)
        offset_y = max(0.0, (label_h - disp_h) / 2.0)

        # map from display coords to original frame coords
        x_disp = float(event.x)
        y_disp = float(event.y)
        x_img = x_disp - offset_x
        y_img = y_disp - offset_y
        if x_img < 0 or y_img < 0 or x_img >= disp_w or y_img >= disp_h:
            return
        base_w = float(self._last_frame_base.shape[1]) if self._last_frame_base is not None else disp_w
        base_h = float(self._last_frame_base.shape[0]) if self._last_frame_base is not None else disp_h
        # use stored mapping from display -> base frame coords
        x_frame = self._disp_ox + x_img * self._disp_sx
        y_frame = self._disp_oy + y_img * self._disp_sy
        x_orig = x_frame * orig_w / base_w
        y_orig = y_frame * orig_h / base_h

        button = {1: "left", 3: "right"}.get(getattr(event, "num", None), str(getattr(event, "num", "?")))
        click_info = {
            "frame": self._last_idx,
            "frame_idx": self._last_idx,
            "phase": phase,
            "button": button,
            "track_id": self.active_track_id,
            "x_disp": x_img,
            "y_disp": y_img,
            # base-frame coordinates (resized frame before view transform)
            "x_base": x_frame,
            "y_base": y_frame,
            "x": x_orig,
            "y": y_orig,
            "x_cm": x_orig / self._current_dpcm if self._current_dpcm else x_orig,
            "y_cm": y_orig / self._current_dpcm if self._current_dpcm else y_orig,
            "dpcm": self._current_dpcm if self._current_dpcm else 1.0,
            "t": time.time(),
        }
        if self._measure_mode:
            # handle measure clicks locally (only on press)
            if phase != "down":
                return
            self._measure_clicks.append(click_info)
            self._handle_measure_click(click_info)
            return
        self.click_events.append(click_info)
        self._publish_event({"type": "click", **click_info})

    def _handle_measure_click(self, click_info: dict) -> None:
        """Process clicks during measure mode using frame/video coords (robust to zoom/pan)."""
        if not self._measure_mode:
            return
        x_base = click_info.get("x_base")
        y_base = click_info.get("y_base")
        x_orig = click_info.get("x")
        y_orig = click_info.get("y")
        if x_base is None or y_base is None:
            return
        if self._measure_phase == "first":
            self._measure_p1 = (x_base, y_base)
            self._measure_p1_orig = (x_orig, y_orig) if x_orig is not None and y_orig is not None else None
            self._measure_p2 = None
            self._measure_p2_orig = None
            self._measure_enter_blocked = False
            self._refresh_measure_frame()
        elif self._measure_phase == "second":
            self._measure_p2 = (x_base, y_base)
            self._measure_p2_orig = (x_orig, y_orig) if x_orig is not None and y_orig is not None else None
            self._measure_enter_blocked = False
            self._refresh_measure_frame()

    def _confirm_measure_point(self) -> None:
        """Handle Enter key during measure mode."""
        if not self._measure_mode:
            return
        if self._measure_enter_blocked:
            return
        if self._measure_phase == "first":
            # advance to second point selection
            self._measure_phase = "second"
            self._measure_enter_blocked = True
            self._measure_prompt = "Please select second point (enter to confirm)"
            self._measure_overlay_cache = None
            self._measure_overlay_sig = None
            self._refresh_measure_frame()
            return
        if self._measure_phase == "second":
            if self._measure_p1 is None or self._measure_p2 is None:
                return
            # Move to length entry phase
            self._measure_phase = "length"
            self._measure_enter_blocked = True
            self._measure_prompt = "Enter the measure's physical length"
            self._measure_overlay_cache = None
            self._measure_overlay_sig = None
            self._open_measure_popup()
            self._refresh_measure_frame()
            return
        if self._measure_phase == "length":
            if not self._measure_popup:
                return
            if self._measure_enter_blocked:
                return
            if not self._validate_measure_popup():
                return
            # Success: close popup and exit measure mode
            self._confirm_measure_popup()

    def _publish_event(self, evt: dict) -> None:
        try:
            self.event_queue.put(evt, timeout=0.1)
        except Exception:
            try:
                # if queue was full, make room and retry once
                _ = self.event_queue.get_nowait()
                self.event_queue.put_nowait(evt)
            except Exception:
                pass

    def _open_measure_popup(self) -> None:
        # destroy existing popup
        self._close_measure_popup()
        if self._measure_p1 is None or self._measure_p2 is None:
            return
        # midpoint in display coords
        def base_to_disp(pt: Tuple[float, float]) -> Tuple[float, float]:
            w = float(self._last_frame_base.shape[1]) if self._last_frame_base is not None else float(self._view_w or 1.0)
            h = float(self._last_frame_base.shape[0]) if self._last_frame_base is not None else float(self._view_h or 1.0)
            vx0 = self._view_x0 or 0.0
            vy0 = self._view_y0 or 0.0
            vw = self._view_w or w
            vh = self._view_h or h
            return (
                (pt[0] - vx0) * (w / vw),
                (pt[1] - vy0) * (h / vh),
            )
        p1_disp = base_to_disp(self._measure_p1)
        p2_disp = base_to_disp(self._measure_p2)
        mid_x = (p1_disp[0] + p2_disp[0]) / 2.0
        mid_y = (p1_disp[1] + p2_disp[1]) / 2.0
        dlg = tk.Toplevel(self.root)
        self._measure_popup = dlg
        dlg.transient(self.root)
        dlg.wm_overrideredirect(True)  # remove window chrome and movement
        dlg.title("Measure")

        # Place near midpoint
        try:
            lx = self.label.winfo_rootx()
            ly = self.label.winfo_rooty()
            popup_w, popup_h = 300, 150
            x = int(lx + mid_x - popup_w / 2)
            y = int(ly + mid_y - popup_h / 2)
            dlg.geometry(f"{popup_w}x{popup_h}+{x}+{y}")
        except Exception:
            dlg.geometry("300x150")

        container = tk.Frame(dlg, padx=8, pady=4)
        container.pack(fill="both", expand=True)

        self._measure_error_var = tk.StringVar(value="")
        err_lbl = tk.Label(container, textvariable=self._measure_error_var, fg="red")
        err_lbl.pack(anchor="w")

        # Mantissa and unit row
        if self._measure_mant_var is None:
            self._measure_mant_var = tk.StringVar(value="")
        if self._measure_unit_var is None:
            self._measure_unit_var = tk.StringVar(value="cm")
        if self._measure_equation_var is None:
            self._measure_equation_var = tk.StringVar(value="")
        row1 = tk.Frame(container)
        row1.pack(fill="x", pady=(2, 2))
        mant_entry = tk.Entry(row1, textvariable=self._measure_mant_var, width=12)
        mant_entry.pack(side="left", padx=(0, 2))
        units = ["km", "m", "cm", "mm", "um", "nm", "in", "ft", "yd", "mile"]
        unit_menu = tk.OptionMenu(row1, self._measure_unit_var, *units, command=lambda _: self._on_measure_changed())
        unit_menu.pack(side="left", padx=(6, 0))
        unit_menu.configure(takefocus=0)

        # Pixels length info + separate DPI entry below
        dpi_row = tk.Frame(container)
        dpi_row.pack(fill="x", pady=(4, 0))
        tk.Label(dpi_row, textvariable=self._measure_equation_var, justify="left").pack(side="left")

        # Dots-per-unit box directly beneath
        dpu_row = tk.Frame(container)
        dpu_row.pack(fill="x", pady=(6, 0))
        if self._measure_dpi_var is None:
            self._measure_dpi_var = tk.StringVar(value="")
        dpi_entry = tk.Entry(dpu_row, textvariable=self._measure_dpi_var, width=10)
        dpi_entry.pack(side="left", padx=(0, 4))
        self._dpi_unit_label = tk.Label(dpu_row, text="px/cm")
        self._dpi_unit_label.pack(side="left")
        self._dpi_equiv_label = tk.Label(dpu_row, text="≈ 0 px/cm")
        self._dpi_equiv_label.pack(side="left", padx=(6, 0))

        # Bind updates
        mant_entry.bind("<FocusIn>", lambda e: self._unblock_measure_enter())
        dpi_entry.bind("<FocusIn>", lambda e: self._unblock_measure_enter())
        mant_entry.bind("<KeyRelease>", lambda e: self._on_measure_changed())
        self._measure_unit_var.trace_add("write", lambda *_: self._on_measure_changed())
        dpi_entry.bind("<KeyRelease>", lambda e: self._on_dpi_changed())

        # init derived values
        self._update_measure_equation_label()
        self._on_dpi_changed()  # will set measure if dpi present, else noop
        self._update_measure_equation_label()
        self._unblock_measure_enter()
        mant_entry.focus_set()
        self._measure_popup.update_idletasks()
        self._measure_popup.lift()

        # Buttons row
        btn_row = tk.Frame(container)
        btn_row.pack(fill="x", pady=(4, 0))
        cancel_btn = tk.Button(btn_row, text="Cancel", command=self._cancel_measure_mode)
        cancel_btn.pack(side="left")
        confirm_btn = tk.Button(btn_row, text="Confirm", command=self._confirm_measure_popup)
        confirm_btn.pack(side="right")

    def _close_measure_popup(self) -> None:
        if self._measure_popup is not None and self._measure_popup.winfo_exists():
            self._measure_popup.destroy()
        self._measure_popup = None
        self._measure_mant_var = None
        self._measure_unit_var = None
        self._measure_dpi_var = None
        self._measure_error_var = None
        self._measure_equation_var = None
        self._restore_focus_to_main()

    def _unblock_measure_enter(self) -> None:
        self._measure_enter_blocked = False

    def _measure_pixel_length(self) -> float:
        p1o, p2o = self._measure_p1_orig, self._measure_p2_orig
        if p1o is not None and p2o is not None:
            dx = p1o[0] - p2o[0]
            dy = p1o[1] - p2o[1]
            return (dx * dx + dy * dy) ** 0.5
        if self._measure_p1 is None or self._measure_p2 is None:
            return 0.0
        dx = self._measure_p1[0] - self._measure_p2[0]
        dy = self._measure_p1[1] - self._measure_p2[1]
        return (dx * dx + dy * dy) ** 0.5

    @staticmethod
    def _fmt_sig(val: float, sig: int = 4) -> str:
        try:
            return f"{float(val):.{sig}g}"
        except Exception:
            return str(val)

    @staticmethod
    def _unit_scale_cm(unit: str) -> float:
        return {
            "km": 100000.0,
            "m": 100.0,
            "cm": 1.0,
            "mm": 0.1,
            "um": 1e-4,
            "nm": 1e-7,
            "in": 2.54,
            "ft": 30.48,
            "yd": 91.44,
            "mile": 160934.4,
        }.get(unit, 1.0)

    def _on_measure_changed(self) -> None:
        if self._measure_updating:
            return
        self._measure_updating = True
        try:
            val = self._parse_measure_value()
            if val is None or val == 0:
                return
            unit = self._measure_unit_var.get() if self._measure_unit_var else "cm"
            unit_scale_cm = self._unit_scale_cm(unit)
            length_unit = val / unit_scale_cm  # user-entered unit length
            if length_unit == 0:
                return
            dpu = self._measure_pixel_length() / length_unit  # pixels per selected unit
            if self._measure_dpi_var is not None:
                self._measure_dpi_var.set(self._fmt_sig(dpu, 4))
        finally:
            self._measure_updating = False
        self._update_measure_equation_label()

    def _on_dpi_changed(self) -> None:
        if self._measure_updating:
            return
        self._measure_updating = True
        try:
            if self._measure_dpi_var is None:
                return
            txt = self._measure_dpi_var.get().strip()
            if not txt:
                return
            dpu = float(txt)  # pixels per selected unit
            if dpu == 0:
                return
            unit = self._measure_unit_var.get() if self._measure_unit_var else "cm"
            unit_scale_cm = self._unit_scale_cm(unit)
            length_unit = self._measure_pixel_length() / dpu  # in selected unit
            length_cm = length_unit * unit_scale_cm
            unit = self._measure_unit_var.get() if self._measure_unit_var else "cm"
            if self._measure_mant_var is not None:
                self._measure_mant_var.set(self._fmt_sig(length_unit, 4))
        except Exception:
            pass
        finally:
            self._measure_updating = False
        self._update_measure_equation_label()

    def _parse_measure_value(self) -> Optional[float]:
        if self._measure_mant_var is None or self._measure_unit_var is None:
            return None
        mant_txt = self._measure_mant_var.get().replace(" ", "")
        if mant_txt == "":
            return None
        unit = self._measure_unit_var.get() if self._measure_unit_var else "cm"
        unit_scale_cm = self._unit_scale_cm(unit)
        try:
            mant = float(mant_txt)
            return mant * unit_scale_cm
        except Exception:
            return None

    def _validate_measure_popup(self) -> bool:
        val = self._parse_measure_value()
        if val is None:
            if self._measure_error_var is not None:
                self._measure_error_var.set("Please only enter numbers")
            return False
        if self._measure_error_var is not None:
            self._measure_error_var.set("")
        return True

    def _confirm_measure_popup(self) -> None:
        """Confirm measure entry, save results, exit measure mode."""
        if not self._validate_measure_popup():
            return
        length_cm = self._parse_measure_value()
        dpi_val = None
        try:
            dpi_val = float(self._measure_dpi_var.get()) if self._measure_dpi_var else None
        except Exception:
            dpi_val = None
        self._measure_result = {
            "length_cm": length_cm,
            "dpi": dpi_val,
            "pixels": self._measure_pixel_length(),
        }
        if length_cm and length_cm > 0:
            self._current_dpcm = self._measure_pixel_length() / length_cm
        else:
            self._current_dpcm = 1.0
        # Enable grid by default after a successful measure
        if self._grid_var is not None:
            self._grid_var.set(True)
        self._show_grid = True
        self._publish_event({"type": "options", "marker_size": self._marker_size, "marker_alpha": self._marker_alpha})
        self._close_measure_popup()
        self._cancel_measure_mode()
        self._restore_focus_to_main()

    def _scale_overlays_for_view(self, overlays: list) -> list:
        """Scale marker sizes/widths based on current zoom; min 1 video pixel."""
        if not overlays:
            return overlays
        zoom = max(1.0, min(self._zoom_scale, 10.0))
        if abs(zoom - 1.0) < 1e-6:
            return overlays
        scaled = []
        for ov in overlays:
            if not isinstance(ov, dict):
                scaled.append(ov)
                continue
            kind = ov.get("kind")
            new_ov = dict(ov)
            if kind == "dot":
                r = ov.get("r", 4)
                new_ov["r"] = max(1, int(round(r / zoom)))
            elif kind == "line":
                w = ov.get("width", 2)
                new_ov["width"] = max(1, int(round(w / zoom)))
            scaled.append(new_ov)
        return scaled

    def _visible_overlays(self, overlays: list) -> list:
        """Apply zoom scaling and hide-only-marker filtering."""
        overlays = self._scale_overlays_for_view(overlays)
        if not self._hide_markers or not overlays:
            return overlays
        try:
            return [ov for ov in overlays if not (isinstance(ov, dict) and ov.get("kind") == "dot")]
        except Exception:
            return overlays

    def _restore_focus_to_main(self) -> None:
        """Ensure keyboard focus returns to main window/label."""
        try:
            self.root.focus_force()
            self.label.focus_set()
        except Exception:
            pass

    def _fit_window_to_viewport(self) -> None:
        """Resize the main window to closely wrap the current viewport once."""
        if self._auto_fit_done or self._photo is None:
            return
        try:
            self._auto_fit_attempts += 1
            self.root.update_idletasks()
            img_w = self._photo.width()
            img_h = self._photo.height()
            dropdown_h = 0
            if self._dropdown_open and self._dropdown_panel is not None and self._dropdown_panel.winfo_ismapped():
                dropdown_h = max(self._dropdown_height, self._dropdown_panel.winfo_height())
            elif self._dropdown_tab is not None and self._dropdown_tab.winfo_ismapped():
                dropdown_h = self._dropdown_tab.winfo_height()
            status_h = self.status.winfo_height() if self.status.winfo_ismapped() else 0
            padding = 6
            win_w = int(img_w + padding * 2)
            win_h = int(img_h + dropdown_h + status_h + padding * 2)
            screen_w = self.root.winfo_screenwidth()
            screen_h = self.root.winfo_screenheight()
            win_w = min(win_w, int(screen_w * 0.95))
            win_h = min(win_h, int(screen_h * 0.95))
            self.root.geometry(f"{win_w}x{win_h}+20+20")
            self.root.update_idletasks()
            actual_h = self.root.winfo_height()
            if abs(actual_h - win_h) <= 8 or self._auto_fit_attempts >= 3:
                self._auto_fit_done = True
        except Exception:
            pass

    # ----------------------- view (zoom/pan) ----------------------- #
    def _apply_view_transform(self, frame: np.ndarray) -> np.ndarray:
        """Apply pan/zoom crop to the frame (frame already includes overlays)."""
        if frame is None:
            return frame
        h, w = frame.shape[:2]
        zoom = max(1.0, min(self._zoom_scale, 10.0))
        # Do not let view exceed full frame when zooming out
        view_w = min(w, w / zoom)
        view_h = min(h, h / zoom)
        cx = w / 2.0 + self._pan_x
        cy = h / 2.0 + self._pan_y
        x0 = max(0.0, min(w - view_w, cx - view_w / 2.0))
        y0 = max(0.0, min(h - view_h, cy - view_h / 2.0))
        self._view_x0, self._view_y0, self._view_w, self._view_h = x0, y0, view_w, view_h
        # store display->frame mapping (base coords)
        self._disp_sx = view_w / w
        self._disp_sy = view_h / h
        self._disp_ox = x0
        self._disp_oy = y0

        x1 = int(x0)
        y1 = int(y0)
        x2 = int(x0 + view_w)
        y2 = int(y0 + view_h)
        crop = frame[y1:y2, x1:x2]
        if crop.shape[0] <= 0 or crop.shape[1] <= 0:
            return frame
        img = Image.fromarray(crop)
        fast = (time.time() - self._last_zoom_ts) < self._zoom_fast_window if self._last_zoom_ts else False
        if fast and cv2 is not None:
            resized = cv2.resize(crop, (w, h), interpolation=cv2.INTER_LINEAR)
            return resized
        resample = Image.Resampling.BILINEAR if fast else Image.Resampling.LANCZOS
        img = img.resize((w, h), resample)
        return np.asarray(img)

    def _apply_grid(self, frame: np.ndarray) -> np.ndarray:
        """Overlay a calibrated grid; tick labels stick to viewport edges."""
        if frame is None or not self._show_grid:
            return frame
        dpcm = self._current_dpcm if self._current_dpcm else 1.0
        if dpcm <= 0:
            return frame
        h, w = frame.shape[:2]
        # view window in base coords
        vx0 = self._view_x0 or 0.0
        vy0 = self._view_y0 or 0.0
        vw = self._view_w or float(w)
        vh = self._view_h or float(h)
        sx = w / vw
        sy = h / vh
        # spacing: choose cm step giving ~80 px in view space
        desired_px = 80.0
        candidates_cm = [0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100]
        spacing_cm = candidates_cm[-1]
        for cm in candidates_cm:
            if cm * dpcm >= desired_px:
                spacing_cm = cm
                break
        step_px = spacing_cm * dpcm  # in base pixels
        base_img = Image.fromarray(frame).convert("RGBA")
        overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        # Sub-grid: when zoomed in so the viewport spans ~3 grid cells or fewer
        sub_start = step_px * 4.0  # start fading in (viewport ~4 cells)
        sub_full = step_px * 3.0   # fully visible (~3 cells)
        fade_factor_w = 0.0
        fade_factor_h = 0.0
        if step_px > 0:
            def fade(vw_or_vh: float) -> float:
                if vw_or_vh >= sub_start:
                    return 0.0
                if vw_or_vh <= sub_full:
                    return 1.0
                return (sub_start - vw_or_vh) / (sub_start - sub_full)
            fade_factor_w = fade(vw)
            fade_factor_h = fade(vh)
        fade_factor = min(fade_factor_w, fade_factor_h)

        if step_px > 0 and fade_factor > 0:
            sub_step = step_px / 5.0
            base_sub_alpha = 0.30 * fade_factor  # overall small-grid opacity with zoom-based fade-in
            feather_px = 40.0  # fade near edges of viewport

            # vertical sub-lines
            sx0 = ((vx0 // sub_step) * sub_step)
            if sx0 < vx0:
                sx0 += sub_step
            sx_pos = sx0
            while sx_pos <= vx0 + vw:
                x_pix = (sx_pos - vx0) * sx
                edge_factor = min(
                    1.0,
                    max(0.0, min(x_pix, w - x_pix) / feather_px),
                )
                alpha = base_sub_alpha * edge_factor
                sub_color = (255, 255, 255, int(255 * alpha))
                draw.line((x_pix, 0, x_pix, h), fill=sub_color, width=1)
                sx_pos += sub_step

            # horizontal sub-lines
            sy0 = ((vy0 // sub_step) * sub_step)
            if sy0 < vy0:
                sy0 += sub_step
            sy_pos = sy0
            while sy_pos <= vy0 + vh:
                y_pix = (sy_pos - vy0) * sy
                edge_factor = min(
                    1.0,
                    max(0.0, min(y_pix, h - y_pix) / feather_px),
                )
                alpha = base_sub_alpha * edge_factor
                sub_color = (255, 255, 255, int(255 * alpha))
                draw.line((0, y_pix, w, y_pix), fill=sub_color, width=1)
                sy_pos += sub_step

        # Main grid stays brighter than the fine sub-grid
        line_color = (255, 255, 255, int(255 * 0.65))
        label_color = (255, 255, 255, 220)

        # vertical lines: iterate visible lines in view coords
        x_start = ((vx0 // step_px) * step_px)
        if x_start < vx0:
            x_start += step_px
        x = x_start
        while x <= vx0 + vw:
            x_pix = (x - vx0) * sx
            draw.line((x_pix, 0, x_pix, h), fill=line_color, width=1)
            # label anchored to bottom edge (x free)
            label_val_cm = x / dpcm
            draw.text((x_pix + 2, h - 16), f"{label_val_cm:.1f} cm", fill=label_color)
            x += step_px

        # horizontal lines
        y_start = ((vy0 // step_px) * step_px)
        if y_start < vy0:
            y_start += step_px
        y = y_start
        while y <= vy0 + vh:
            y_pix = (y - vy0) * sy
            draw.line((0, y_pix, w, y_pix), fill=line_color, width=1)
            # label anchored to left edge (y free)
            label_val_cm = y / dpcm
            draw.text((2, y_pix + 2), f"{label_val_cm:.1f} cm", fill=label_color)
            y += step_px
        composed = Image.alpha_composite(base_img, overlay)
        return np.asarray(composed.convert("RGB"))

    def _maybe_settle_zoom(self, seq: int) -> None:
        """After scroll burst, redraw with high-quality resample if unchanged."""
        if seq != self._zoom_seq:
            return
        if (time.time() - self._last_zoom_ts) < self._zoom_fast_window:
            return
        # redraw current frame with premium resample
        self._redraw_current_frame(apply_measure=self._measure_mode)

    def _request_zoom_redraw(self) -> None:
        """Coalesce multiple wheel events into a single redraw per loop."""
        self._pending_zoom_redraw = True
        if self._zoom_redraw_scheduled:
            return
        self._zoom_redraw_scheduled = True
        self.root.after_idle(self._perform_zoom_redraw)

    def _perform_zoom_redraw(self) -> None:
        self._zoom_redraw_scheduled = False
        if not self._pending_zoom_redraw:
            return
        self._pending_zoom_redraw = False
        self._redraw_current_frame(apply_measure=self._measure_mode)

    def _on_zoom(self, event: tk.Event, delta: Optional[int] = None) -> None:
        """Mouse wheel zoom, centered on cursor."""
        if self._last_frame_base is None:
            return
        if delta is None:
            delta = event.delta if hasattr(event, "delta") else 0
        if delta == 0:
            return
        # smoother scaling: map wheel delta (typically +/-120) to a fractional step; higher gain = faster
        gain = self._zoom_speed_gain if self._zoom_speed_gain else 300.0
        step = (delta / 120.0) * (gain / 2000.0)
        step = max(-0.8, min(0.8, step))  # clamp runaway
        factor = 1.0 + step
        if factor <= 0.02:  # defensive floor
            factor = 0.02
        # Prevent hidden zoom values below full-frame (deadzone on zoom-in)
        new_zoom = max(1.0, min(self._zoom_scale * factor, 8.0))

        # cursor position relative to image
        disp_w = float(self._photo.width()) if self._photo is not None else self._last_display_shape[1]
        disp_h = float(self._photo.height()) if self._photo is not None else self._last_display_shape[0]
        label_w = max(1.0, float(self.label.winfo_width()))
        label_h = max(1.0, float(self.label.winfo_height()))
        offset_x = max(0.0, (label_w - disp_w) / 2.0)
        offset_y = max(0.0, (label_h - disp_h) / 2.0)
        x_img = float(event.x) - offset_x
        y_img = float(event.y) - offset_y
        if x_img < 0 or y_img < 0 or x_img >= disp_w or y_img >= disp_h:
            x_img = disp_w / 2.0
            y_img = disp_h / 2.0

        # map cursor to frame coords before zoom
        view_w = self._view_w if self._view_w is not None else disp_w
        view_h = self._view_h if self._view_h is not None else disp_h
        view_x0 = self._view_x0 if self._view_x0 is not None else 0.0
        view_y0 = self._view_y0 if self._view_y0 is not None else 0.0
        base_w = self._last_frame_base.shape[1]
        base_h = self._last_frame_base.shape[0]
        frame_x = view_x0 + x_img * view_w / disp_w
        frame_y = view_y0 + y_img * view_h / disp_h

        # apply zoom and adjust pan so the cursor stays on same frame point
        self._zoom_scale = new_zoom
        new_view_w = base_w / new_zoom
        new_view_h = base_h / new_zoom
        new_x0 = frame_x - x_img * new_view_w / disp_w
        new_y0 = frame_y - y_img * new_view_h / disp_h
        # convert to pan offsets (center-based)
        self._pan_x = (new_x0 + new_view_w / 2.0) - base_w / 2.0
        self._pan_y = (new_y0 + new_view_h / 2.0) - base_h / 2.0

        self._last_zoom_ts = time.time()
        self._zoom_seq += 1
        seq = self._zoom_seq
        settle_ms = int(self._zoom_fast_window * 1000) + 30
        self.root.after(settle_ms, lambda s=seq: self._maybe_settle_zoom(s))
        self._request_zoom_redraw()
        self._update_minimap()

    def _on_motion_pan(self, event: tk.Event) -> None:
        """Shift + mouse move pans: displacement from anchor controls pan speed."""
        if self._last_frame_base is None:
            return
        shift_pressed = bool(event.state & 0x0001)
        # sizes in display space
        base_w = float(self._last_frame_base.shape[1])
        base_h = float(self._last_frame_base.shape[0])
        disp_w = float(self._photo.width()) if self._photo is not None else base_w
        disp_h = float(self._photo.height()) if self._photo is not None else base_h
        label_w = max(1.0, float(self.label.winfo_width()))
        label_h = max(1.0, float(self.label.winfo_height()))
        offset_x = max(0.0, (label_w - disp_w) / 2.0)
        offset_y = max(0.0, (label_h - disp_h) / 2.0)
        x_img = float(event.x) - offset_x
        y_img = float(event.y) - offset_y

        if not shift_pressed:
            self._clear_pan_anchor()
            return

        if self._pan_anchor is None:
            self._pan_anchor = (x_img, y_img)
            self._pan_cursor = (x_img, y_img)
            self._show_anchor_at(x_img, y_img, offset_x, offset_y)
            return

        self._pan_cursor = (x_img, y_img)
        self._show_anchor_at(self._pan_anchor[0], self._pan_anchor[1], offset_x, offset_y)
        self._apply_continuous_pan(redraw=True)
        self._update_anchor_canvas()

    def _clear_pan_anchor(self) -> None:
        self._pan_anchor = None
        self._pan_cursor = None
        self._hide_anchor_popup()

    def _show_anchor_at(self, x_img: float, y_img: float, offset_x: float, offset_y: float) -> None:
        lx = self.label.winfo_rootx()
        ly = self.label.winfo_rooty()
        x_scr = lx + offset_x + x_img
        y_scr = ly + offset_y + y_img
        popup = self._anchor_popup
        if popup is None or not popup.winfo_exists():
            popup = tk.Toplevel(self.root)
            popup.overrideredirect(True)
            popup.attributes("-topmost", True)
            canvas = tk.Canvas(popup, width=16, height=16, highlightthickness=0, bd=0, bg=self.root.cget("background"))
            canvas.pack()
            canvas.create_oval(2, 2, 14, 14, outline="#00ff00", width=2)
            self._anchor_popup = popup
        popup.geometry(f"+{int(x_scr - 8)}+{int(y_scr - 8)}")
        popup.lift()

    def _update_anchor_canvas(self) -> None:
        if self._pan_anchor is None:
            self._hide_anchor_popup()
            return
        if self._photo is None or self._anchor_popup is None or not self._anchor_popup.winfo_exists():
            return
        label_w = max(1.0, float(self.label.winfo_width()))
        label_h = max(1.0, float(self.label.winfo_height()))
        disp_w = float(self._photo.width())
        disp_h = float(self._photo.height())
        offset_x = max(0.0, (label_w - disp_w) / 2.0)
        offset_y = max(0.0, (label_h - disp_h) / 2.0)
        x_img, y_img = self._pan_anchor
        lx = self.label.winfo_rootx()
        ly = self.label.winfo_rooty()
        x_scr = lx + offset_x + x_img
        y_scr = ly + offset_y + y_img
        self._anchor_popup.geometry(f"+{int(x_scr - 8)}+{int(y_scr - 8)}")
        self._anchor_popup.lift()

    def _hide_anchor_popup(self) -> None:
        if self._anchor_popup is not None and self._anchor_popup.winfo_exists():
            self._anchor_popup.destroy()
        self._anchor_popup = None

    def _apply_continuous_pan(self, redraw: bool = False) -> None:
        """Apply pan based on anchor->cursor displacement; optionally redraw."""
        if self._pan_anchor is None or self._pan_cursor is None or self._photo is None or self._last_frame_base is None:
            return
        disp_w = float(self._photo.width())
        disp_h = float(self._photo.height())
        anchor_x, anchor_y = self._pan_anchor
        cur_x, cur_y = self._pan_cursor
        dx_img = cur_x - anchor_x
        dy_img = cur_y - anchor_y
        view_w = self._view_w if self._view_w is not None else disp_w
        view_h = self._view_h if self._view_h is not None else disp_h
        # use a gentle ramp: small displacements give tiny pan; larger give faster pan
        norm_dx = dx_img / disp_w
        norm_dy = dy_img / disp_h
        gain = self._pan_speed_gain
        def ramp(v: float) -> float:
            return (abs(v) ** 1.5) * (1 if v >= 0 else -1)
        self._pan_x += ramp(norm_dx) * view_w * gain
        self._pan_y += ramp(norm_dy) * view_h * gain
        if redraw:
            self._redraw_current_frame(apply_measure=self._measure_mode)
            self._update_minimap()
            self._update_anchor_canvas()

    def _update_minimap(self) -> None:
        """Draw minimap with video bounds and current viewport."""
        if self._minimap_canvas is None or self._last_frame_base is None:
            return
        c = self._minimap_canvas
        c.delete("all")
        mw, mh = self._minimap_size
        margin = self._minimap_margin
        base_h, base_w = self._last_frame_base.shape[:2]
        if base_w <= 0 or base_h <= 0:
            return
        scale = min((mw - 2 * margin) / base_w, (mh - 2 * margin) / base_h)
        vw = base_w * scale
        vh = base_h * scale
        x0 = margin + (mw - 2 * margin - vw) / 2
        y0 = margin + (mh - 2 * margin - vh) / 2
        # video boundary
        c.create_rectangle(x0, y0, x0 + vw, y0 + vh, outline="#bbbbbb", width=1)
        # viewport
        view_w = self._view_w if self._view_w is not None else base_w
        view_h = self._view_h if self._view_h is not None else base_h
        view_x0 = self._view_x0 if self._view_x0 is not None else 0.0
        view_y0 = self._view_y0 if self._view_y0 is not None else 0.0
        vx0 = x0 + view_x0 * scale
        vy0 = y0 + view_y0 * scale
        vx1 = vx0 + view_w * scale
        vy1 = vy0 + view_h * scale
        c.create_rectangle(vx0, vy0, vx1, vy1, outline="#4caf50", width=2)

    # ----------------------- track management ----------------------- #
    def _init_tracks(self) -> None:
        self.tracks = []
        self._track_var = tk.StringVar(master=self.root, value=self.active_track_id)
        self._add_track("track1")

    def _init_styles(self) -> None:
        """Configure a dark, borderless style for dropdown controls."""
        try:
            self._style = ttk.Style(self.root)
            try:
                self._style.theme_use("clam")
            except Exception:
                pass
            self._style.configure(
                "Dark.TButton",
                background="#3b3b3b",
                foreground="white",
                borderwidth=1,
                bordercolor="#3b3b3b",
                lightcolor="#3b3b3b",
                darkcolor="#3b3b3b",
                focusthickness=0,
                focuscolor=self.root.cget("background"),
                padding=(8, 4),
                relief="flat",
            )
            self._style.map(
                "Dark.TButton",
                background=[("active", "#555555"), ("pressed", "#555555")],
                foreground=[("disabled", "#bbbbbb")],
                relief=[("pressed", "flat"), ("active", "flat")],
            )
            self._style.configure(
                "Dark.TMenubutton",
                background="#3b3b3b",
                foreground="white",
                borderwidth=1,
                bordercolor="#3b3b3b",
                lightcolor="#3b3b3b",
                darkcolor="#3b3b3b",
                focusthickness=0,
                padding=(8, 4),
                relief="flat",
            )
            self._style.map(
                "Dark.TMenubutton",
                background=[("active", "#555555"), ("pressed", "#555555")],
                foreground=[("disabled", "#bbbbbb")],
                relief=[("pressed", "flat"), ("active", "flat")],
            )
            self._style.configure(
                "Thin.Horizontal.TProgressbar",
                troughcolor="#1e1e1e",
                background="#4caf50",
                bordercolor="#1e1e1e",
                lightcolor="#6edc82",
                darkcolor="#4caf50",
                thickness=6,
            )
        except Exception:
            pass

    def _style_button(self, btn: tk.Widget) -> None:
        """Apply dark, flat styling to a ttk or Tk button-like widget."""
        try:
            if isinstance(btn, ttk.Widget):
                btn.configure(style="Dark.TButton")
            else:
                btn.configure(
                    bg="#3b3b3b",
                    fg="white",
                    activebackground="#555555",
                    activeforeground="white",
                    highlightbackground="#3b3b3b",
                    highlightcolor="#3b3b3b",
                    highlightthickness=0,
                    bd=0,
                    relief="flat",
                    padx=6,
                    pady=4,
                )
        except Exception:
            pass

    def _slider_to_onion_frames(self, slider_val: int) -> int:
        """Map slider (0-100) to onion frame count on a log scale: 0->0, 100->1000."""
        try:
            import math

            s = max(0.0, min(100.0, float(slider_val))) / 100.0
            if s <= 0.0:
                return 0
            frames = int(round(10 ** (s * math.log10(1000.0))))
            return max(0, min(1000, frames))
        except Exception:
            return self._onion_frames

    def _onion_frames_to_slider(self, frames: int) -> int:
        """Inverse mapping for the log slider: frames -> slider (0-100)."""
        try:
            if frames <= 0:
                return 0
            import math

            f = max(1, min(1000, int(frames)))
            s = math.log10(f) / math.log10(1000.0)
            return int(round(max(0.0, min(1.0, s)) * 100))
        except Exception:
            return 50

    def _add_track(self, track_id: Optional[str] = None) -> None:
        tid = track_id or f"track{len(self.tracks)+1}"
        color = self._track_colors[(len(self.tracks)) % len(self._track_colors)]
        self.tracks.append({"id": tid, "color": color})
        self.active_track_id = tid
        if self._track_var is not None:
            self._track_var.set(tid)
        # inform worker
        self._publish_event({"type": "track_meta", "track_id": tid, "color": color})
        # refresh dropdown UI if present
        self._refresh_track_selector()

    def _set_active_track(self, tid: str) -> None:
        for t in self.tracks:
            if t["id"] == tid:
                self.active_track_id = tid
                return

    def _refresh_track_selector(self) -> None:
        if not hasattr(self, "_track_select_menu"):
            return
        menu = self._track_select_menu["menu"]
        menu.delete(0, "end")
        for t in self.tracks:
            menu.add_command(label=t["id"], command=lambda val=t["id"]: self._on_track_chosen(val))
        self._track_select_menu.configure(textvariable=self._track_var)

    def _on_track_chosen(self, tid: str) -> None:
        self._set_active_track(tid)
        if self._track_var is not None:
            self._track_var.set(tid)

    # ----------------------- dropdown menu ----------------------- #
    def _init_dropdown_menu(self) -> None:
        """Create a top-center tab that toggles a small dropdown panel."""
        tab = ttk.Button(
            self.label,
            text="Menu ▼",
            command=self._toggle_dropdown,
            takefocus=0,
            style="Dark.TButton",
        )
        tab.place(relx=0.5, y=-1, anchor="n")
        self._dropdown_tab = tab
        # Panel is created lazily when first opened

    def _toggle_dropdown(self) -> None:
        if self._dropdown_animating:
            return
        if self._dropdown_open:
            self._collapse_dropdown()
        else:
            self._expand_dropdown()

    def _toggle_hide_markers(self) -> None:
        self._hide_markers = not self._hide_markers
        if self._hide_markers_btn is not None:
            self._hide_markers_btn.configure(text="Show points" if self._hide_markers else "Hide points")
        self._redraw_current_frame(apply_measure=self._measure_mode)

    def _toggle_hide_ui(self) -> None:
        if self._hide_ui:
            return
        self._hide_ui = True
        # hide status and minimap and dropdown tab/panel
        try:
            self.status.pack_forget()
        except Exception:
            pass
        try:
            if self._minimap_canvas is not None:
                self._minimap_canvas.place_forget()
        except Exception:
            pass
        try:
            if self._dropdown_panel is not None:
                self._dropdown_panel.place_forget()
        except Exception:
            pass
        try:
            if self._dropdown_tab is not None:
                self._dropdown_tab.place_forget()
        except Exception:
            pass
        self._destroy_dropdown_hide_tab()
        # bind one-shot restore on next click
        self._ui_restore_binding = self.label.bind("<Button-1>", lambda e: self._restore_ui_on_click(), add="+")
        self._suppress_click_once = True
        if self._hide_ui_btn is not None:
            self._hide_ui_btn.configure(text="UI hidden (click video to restore)")

    def _restore_ui_on_click(self) -> None:
        if not self._hide_ui:
            return
        self._hide_ui = False
        try:
            self.status.pack(fill="x")
        except Exception:
            pass
        try:
            if self._minimap_canvas is not None:
                self._minimap_canvas.place(relx=1.0, y=6, anchor="ne")
        except Exception:
            pass
        try:
            if self._dropdown_tab is not None:
                self._dropdown_tab.place(relx=0.5, y=-1, anchor="n")
        except Exception:
            pass
        # reset dropdown state and reopen automatically
        self._dropdown_open = False
        self._dropdown_animating = False
        try:
            self._expand_dropdown()
        except Exception:
            pass
        if self._hide_ui_btn is not None:
            self._hide_ui_btn.configure(text="Hide UI")
        # unbind restore handler
        if self._ui_restore_binding:
            try:
                self.label.unbind("<Button-1>", self._ui_restore_binding)
            except Exception:
                pass
            self._ui_restore_binding = None
        self._redraw_current_frame(apply_measure=self._measure_mode)

    def _ensure_dropdown_panel(self) -> tk.Frame:
        if self._dropdown_panel is None:
            panel = tk.Frame(self.label, bg="#2b2b2b", bd=1, relief="ridge")
            panel.propagate(False)  # honor explicit height during animation
            try:
                panel.tk_setPalette(background="#2b2b2b", foreground="white", activeBackground="#555555", activeForeground="white")
            except Exception:
                pass
            try:
                panel.option_add("*Background", "#2b2b2b")
                panel.option_add("*foreground", "white")
            except Exception:
                pass
            # sample content
            tk.Label(panel, text="Tracks", fg="white", bg="#2b2b2b").pack(padx=6, pady=(4, 2), anchor="w")
            track_row = tk.Frame(panel, bg="#2b2b2b")
            track_row.pack(fill="x", padx=6, pady=(0, 4))
            self._controls_row = track_row
            if self._track_var is None:
                self._track_var = tk.StringVar(value=self.active_track_id)
            selector = ttk.OptionMenu(track_row, self._track_var, self._track_var.get(), *[t["id"] for t in self.tracks], command=self._on_track_chosen)
            selector.configure(style="Dark.TMenubutton", takefocus=0)
            try:
                selector["menu"].config(bg="#3b3b3b", fg="white", activebackground="#555555", activeforeground="white", bd=0, relief="flat")
            except Exception:
                pass
            selector.pack(side="left")
            self._track_select_menu = selector
            for text, cmd in [
                ("New track", lambda: self._add_track(None)),
                ("Measure", self._enter_measure_mode),
                ("Save…", self._open_save_dialog),
                ("Options", self._open_options_panel),
            ]:
                btn = ttk.Button(track_row, text=text, command=cmd, takefocus=0, style="Dark.TButton")
                btn.pack(side="left", padx=(6, 0))
            self._dropdown_panel = panel
        else:
            panel = self._dropdown_panel

        self._ensure_visibility_buttons(panel)
        return panel

    def _ensure_visibility_buttons(self, panel: tk.Frame) -> None:
        """Ensure hide/show buttons exist in the dropdown, even if panel was built earlier."""
        if panel is None:
            return
        # if already created, nothing to do
        if self._hide_markers_btn is not None and self._hide_ui_btn is not None:
            return
        row = self._controls_row
        if row is None or not row.winfo_exists():
            row = tk.Frame(panel, bg="#2b2b2b")
            row.pack(fill="x", padx=6, pady=(0, 4))
            self._controls_row = row
        self._hide_markers_btn = ttk.Button(row, text="Hide points", command=self._toggle_hide_markers, takefocus=0, style="Dark.TButton")
        self._hide_markers_btn.pack(side="left", padx=(6, 0))
        self._hide_ui_btn = ttk.Button(row, text="Hide UI", command=self._toggle_hide_ui, takefocus=0, style="Dark.TButton")
        self._hide_ui_btn.pack(side="left", padx=(6, 0))

    def _expand_dropdown(self) -> None:
        panel = self._ensure_dropdown_panel()
        # Adjust target height to fit current content so new buttons are visible
        try:
            panel.update_idletasks()
            req_h = panel.winfo_reqheight()
            if req_h and req_h > 0:
                # Fit content snugly but keep a sensible minimum height for legibility.
                self._dropdown_target_height = max(64, req_h + 4)
            else:
                self._dropdown_target_height = 64
        except Exception:
            pass
        panel.place(relx=0.5, y=0, anchor="n", width=self.label.winfo_width(), height=0)
        self._dropdown_open = True
        self._dropdown_animating = True
        self._animate_dropdown(height_start=0, height_end=self._dropdown_target_height, step=12)
        self._show_dropdown_hide_tab()

    def _collapse_dropdown(self) -> None:
        if self._dropdown_panel is None:
            self._dropdown_open = False
            self._destroy_dropdown_hide_tab()
            return
        self._dropdown_animating = True
        self._animate_dropdown(height_start=self._dropdown_height, height_end=0, step=-12)

    def _animate_dropdown(self, height_start: int, height_end: int, step: int) -> None:
        self._dropdown_height = height_start

        def tick(h: int) -> None:
            self._dropdown_height = h
            if self._dropdown_panel is not None:
                self._dropdown_panel.place(relx=0.5, y=0, anchor="n", width=self.label.winfo_width(), height=max(0, h))
            self._position_dropdown_hide_tab()
            if (step > 0 and h >= height_end) or (step < 0 and h <= height_end):
                # finalize
                if self._dropdown_panel is not None:
                    if height_end == 0:
                        self._dropdown_panel.place_forget()
                self._dropdown_animating = False
                self._dropdown_open = height_end > 0
                if height_end == 0:
                    self._destroy_dropdown_hide_tab()
                return
            self.root.after(15, lambda: tick(h + step))

        tick(height_start + step)

    def _show_dropdown_hide_tab(self) -> None:
        if self._dropdown_hide_popup is not None and self._dropdown_hide_popup.winfo_exists():
            return
        popup = tk.Toplevel(self.root)
        popup.overrideredirect(True)
        popup.transient(self.root)
        popup.attributes("-topmost", True)
        btn = ttk.Button(
            popup,
            text="▲ Hide",
            command=self._collapse_dropdown,
            takefocus=0,
            style="Dark.TButton",
        )
        btn.pack(padx=2, pady=2)
        self._dropdown_hide_popup = popup
        self._position_dropdown_hide_tab()

    def _position_dropdown_hide_tab(self) -> None:
        if self._dropdown_hide_popup is None or not self._dropdown_hide_popup.winfo_exists():
            return
        if self._dropdown_panel is None or not self._dropdown_open:
            return
        try:
            lx = self.label.winfo_rootx()
            ly = self.label.winfo_rooty()
            width = self.label.winfo_width()
            popup_w = self._dropdown_hide_popup.winfo_width() or 70
            x = int(lx + (width - popup_w) / 2)
            y = int(ly + self._dropdown_height)
            self._dropdown_hide_popup.geometry(f"+{x}+{y}")
        except Exception:
            pass

    def _destroy_dropdown_hide_tab(self) -> None:
        if self._dropdown_hide_popup is not None and self._dropdown_hide_popup.winfo_exists():
            self._dropdown_hide_popup.destroy()
        self._dropdown_hide_popup = None

    # ----------------------- options panel ----------------------- #
    def _open_options_panel(self) -> None:
        """Open or focus the options panel (marker styling)."""
        if self._options_win is not None and self._options_win.winfo_exists():
            self._options_win.lift()
            self._options_win.focus_force()
            return
        win = tk.Toplevel(self.root)
        self._options_win = win
        win.title("Options")
        win.transient(self.root)
        win.geometry("260x180")
        win.protocol("WM_DELETE_WINDOW", self._destroy_options_panel)

        container = tk.Frame(win, padx=10, pady=10)
        container.pack(fill="both", expand=True)

        tk.Label(container, text="Marker size (px)").pack(anchor="w")
        if self._marker_size_var is None:
            self._marker_size_var = tk.IntVar(value=self._marker_size)
        size_scale = tk.Scale(
            container,
            from_=2,
            to=20,
            orient="horizontal",
            resolution=1,
            variable=self._marker_size_var,
            command=lambda _v=None: self._on_options_changed(),
            width=10,
        )
        size_scale.pack(fill="x", pady=(0, 10))

        tk.Label(container, text="Marker alpha (0-1)").pack(anchor="w")
        if self._marker_alpha_var is None:
            self._marker_alpha_var = tk.DoubleVar(value=self._marker_alpha)
        alpha_scale = tk.Scale(
            container,
            from_=0.1,
            to=1.0,
            orient="horizontal",
            resolution=0.05,
            variable=self._marker_alpha_var,
            command=lambda _v=None: self._on_options_changed(),
            width=10,
        )
        alpha_scale.pack(fill="x")

        # Zoom speed (sensitivity)
        tk.Label(container, text="Zoom speed (higher = faster)").pack(anchor="w", pady=(10, 0))
        if self._zoom_speed_var is None:
            self._zoom_speed_var = tk.DoubleVar(value=self._zoom_speed_gain)
        zoom_scale = tk.Scale(
            container,
            from_=100,   # slower end
            to=1000,    # faster end
            orient="horizontal",
            resolution=50,
            variable=self._zoom_speed_var,
            command=lambda _v=None: self._on_zoom_speed_changed(),
            width=10,
        )
        zoom_scale.pack(fill="x")

        # Pan speed (sensitivity)
        tk.Label(container, text="Pan speed (higher = faster)").pack(anchor="w", pady=(10, 0))
        if self._pan_speed_var is None:
            self._pan_speed_var = tk.DoubleVar(value=self._pan_speed_gain)
        pan_scale = tk.Scale(
            container,
            from_=0.2,   # slower
            to=1.5,     # faster
            resolution=0.05,
            orient="horizontal",
            variable=self._pan_speed_var,
            command=lambda _v=None: self._on_pan_speed_changed(),
            width=10,
        )
        pan_scale.pack(fill="x")

        # Onionskin duration (log slider: mid ~10, max ~1000)
        tk.Label(container, text="Onionskin duration (frames, log)").pack(anchor="w", pady=(10, 0))
        if self._onion_slider_var is None:
            self._onion_slider_var = tk.IntVar(value=self._onion_frames_to_slider(self._onion_frames))
        onion_scale = tk.Scale(
            container,
            from_=0,
            to=100,
            orient="horizontal",
            resolution=1,
            variable=self._onion_slider_var,
            command=lambda _v=None: self._on_options_changed(),
            width=10,
        )
        onion_scale.pack(fill="x")

        # Grid overlay toggle
        if self._grid_var is None:
            self._grid_var = tk.BooleanVar(value=self._show_grid)
        grid_chk = tk.Checkbutton(
            container,
            text="Show calibrated grid",
            variable=self._grid_var,
            command=self._on_options_changed,
        )
        grid_chk.pack(anchor="w", pady=(10, 0))

        # ensure worker sync when panel opens
        self._on_options_changed()

    def _destroy_options_panel(self) -> None:
        if self._options_win is not None and self._options_win.winfo_exists():
            self._options_win.destroy()
        self._options_win = None

    def _on_options_changed(self) -> None:
        """Handle changes to marker size/alpha and notify worker."""
        try:
            size_val = self._marker_size_var.get() if self._marker_size_var is not None else self._marker_size
        except Exception:
            size_val = self._marker_size
        try:
            alpha_val = self._marker_alpha_var.get() if self._marker_alpha_var is not None else self._marker_alpha
        except Exception:
            alpha_val = self._marker_alpha
        try:
            slider_val = self._onion_slider_var.get() if self._onion_slider_var is not None else self._onion_frames_to_slider(self._onion_frames)
            onion_val = self._slider_to_onion_frames(slider_val)
        except Exception:
            onion_val = self._onion_frames
        size_val = max(1, min(40, int(size_val)))
        alpha_val = max(0.05, min(1.0, float(alpha_val)))
        onion_val = max(0, min(1000, int(onion_val)))
        show_grid = bool(self._grid_var.get()) if self._grid_var is not None else self._show_grid
        if (
            size_val == self._marker_size
            and abs(alpha_val - self._marker_alpha) < 1e-4
            and show_grid == self._show_grid
            and onion_val == self._onion_frames
        ):
            return
        self._marker_size = size_val
        self._marker_alpha = alpha_val
        self._show_grid = show_grid
        self._onion_frames = onion_val
        self._publish_event(
            {
                "type": "options",
                "marker_size": self._marker_size,
                "marker_alpha": self._marker_alpha,
                "onion_frames": self._onion_frames,
            }
        )
        # redraw current frame to apply grid toggle immediately
        self._redraw_current_frame(apply_measure=self._measure_mode)

    def _on_zoom_speed_changed(self) -> None:
        try:
            val = float(self._zoom_speed_var.get()) if self._zoom_speed_var is not None else self._zoom_speed_gain
        except Exception:
            val = self._zoom_speed_gain
        val = max(50.0, min(2000.0, val))
        self._zoom_speed_gain = val

    def _on_pan_speed_changed(self) -> None:
        try:
            val = float(self._pan_speed_var.get()) if self._pan_speed_var is not None else self._pan_speed_gain
        except Exception:
            val = self._pan_speed_gain
        val = max(0.05, min(3.0, val))
        self._pan_speed_gain = val

    def _update_measure_equation_label(self) -> None:
        if self._measure_equation_var is None:
            return
        pix_len = self._measure_pixel_length()
        measure_val_cm = self._parse_measure_value()
        unit = self._measure_unit_var.get() if self._measure_unit_var else "cm"
        pix_disp = f"{int(round(pix_len))}"  # whole pixels for display
        if measure_val_cm is None:
            text = f"{pix_disp} px ÷ [enter length] {unit} ≈"
        else:
            measure_unit_val = measure_val_cm / self._unit_scale_cm(unit)
            text = f"{pix_disp} px ÷ {self._fmt_sig(measure_unit_val,3)} {unit} ≈"
        self._measure_equation_var.set(text)
        # update dpi unit label
        if hasattr(self, "_dpi_unit_label") and self._dpi_unit_label:
            self._dpi_unit_label.config(text=f"px/{unit}")
        if hasattr(self, "_dpi_equiv_label") and self._dpi_equiv_label:
            try:
                dpcm_val = self._measure_pixel_length() / measure_val_cm if measure_val_cm else 0
            except Exception:
                dpcm_val = 0
            self._dpi_equiv_label.config(text=f"≈ {self._fmt_sig(dpcm_val,4)} px/cm")

    # ----------------------- progress bar ----------------------- #
    def _update_progress(self, idx: Optional[int]) -> None:
        """Update the thin progress bar with the current frame index."""
        if not hasattr(self, "progress") or self.progress is None or idx is None:
            return
        try:
            self.progress["maximum"] = self._progress_max
            self.progress["value"] = max(0, min(self._progress_max, int(idx)))
        except Exception:
            pass


def main() -> None:
    parser = argparse.ArgumentParser(description="Minimal Tk live video viewer")
    parser.add_argument("video", type=str, help="Path to video file")
    parser.add_argument("--capacity", type=int, default=120, help="Ring buffer capacity")
    parser.add_argument("--step", type=int, default=1, help="Frame stride (1 = every frame)")
    parser.add_argument("--refresh-ms", type=int, default=33, help="UI refresh interval in ms")
    parser.add_argument("--width", type=int, default=None, help="Display width (px); preserves aspect ratio")
    args = parser.parse_args()

    viewer = LiveViewer(
        video_path=Path(args.video),
        capacity=args.capacity,
        step=args.step,
        refresh_ms=args.refresh_ms,
        width=args.width,
    )
    viewer.start()


if __name__ == "__main__":
    main()
