"""
Simple plotting helpers for lab 2 point-tracking data.
"""

from pathlib import Path
import sys
from typing import Dict, Optional, List, Tuple, Union
import hashlib
import itertools
import math
import json
import pickle
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import MeanShift, estimate_bandwidth

# Hold references to non-blocking figures so GUIs stay open while code continues
_PLOT_FIGURES: List[plt.Figure] = []
_EXPLORATORY_CACHE_VERSION = 1
_SEED_DEBUG_STANDARD_BREAKS = np.asarray(
    [
        3.733028,
        15.02402,
        21.821337,
        25.320343,
        44.712543,
        47.622467,
        62.295737,
        72.562363,
        82.502347,
    ],
    dtype=float,
)
_SEED_DEBUG_MATCH_MARGIN = 0.5


def _display_nonblocking(fig: plt.Figure, track_id: Optional[str] = None) -> Optional[Path]:
    """
    Try to display `fig` in a non-blocking way. If the matplotlib backend
    is non-GUI (e.g. 'agg') or display fails, save the figure to `./plots/`
    and return the path.

    This helper enables interactive mode, draws the canvas, and pauses
    briefly to allow GUI event loops to update. It appends the figure to
    `_PLOT_FIGURES` to avoid garbage collection closing the window.
    """
    try:
        import matplotlib as mpl

        backend = mpl.get_backend().lower()
    except Exception:
        backend = "unknown"

    try:
        # Prefer interactive display when possible
        plt.ion()
        fig.canvas.draw()
        plt.show(block=False)
        # pause a bit to let GUI event loop process events
        plt.pause(0.1)
        _PLOT_FIGURES.append(fig)
        return None
    except Exception:
        # fallback to saving if display failed or backend has no GUI
        try:
            out_dir = Path.cwd() / "plots"
            out_dir.mkdir(parents=True, exist_ok=True)
            name = (track_id or "track")
            fname = out_dir / f"{name}_{int(time.time() * 1000)}.png"
            fig.savefig(str(fname), dpi=150)
            print(f"[plot] saved non-interactive figure to {fname}")
            return fname
        except Exception:
            return None



# Allow running as a standalone script (not as a package).
CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from data_extractor import (
    DEFAULT_DATA_DIR,
    tracks_from_file,
    tracks_from_folder,
    REFINED_DATA_DIR,
    split_by_track_id,
    split_refined_by_track,
)


def _jsonify_value(value: object) -> object:
    """
    Convert numpy/path-heavy structures into JSON-serializable builtins.
    """
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonify_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonify_value(item) for item in value]
    return value


def _append_fit_trace_event(
    trace: Optional[List[Dict[str, object]]],
    event: str,
    **fields: object,
) -> None:
    """
    Append one structured trace record if trace collection is enabled.
    """
    if trace is None:
        return
    record = {"event": str(event)}
    for key, value in fields.items():
        record[str(key)] = _jsonify_value(value)
    trace.append(record)


def _resolve_tracks_input(
    tracks_input: Union[Dict[str, np.ndarray], str, Path],
) -> Tuple[Dict[str, np.ndarray], Dict[str, object]]:
    """
    Accept either an in-memory track dict or a CSV path and return the normalized
    track payload plus source metadata.
    """
    if isinstance(tracks_input, dict):
        return tracks_input, {"kind": "in_memory"}

    source_path = Path(tracks_input).expanduser().resolve()
    if not source_path.is_file():
        raise FileNotFoundError(f"{source_path} is not a readable file")

    df = pd.read_csv(source_path)
    columns = set(df.columns)
    if {"track_id", "time_sec", "y_px"}.issubset(columns):
        if {"x_px", "y_px"}.issubset(columns):
            grouped = split_by_track_id(df)
            tracks_data: Dict[str, np.ndarray] = {}
            for track_id, payload in grouped.items():
                time_sec = np.asarray(payload["time_sec"], dtype=float)
                y_px = np.asarray(payload["xydata_px"][1], dtype=float)
                tracks_data[track_id] = np.vstack((time_sec, y_px))
            source_kind = "raw_csv"
        else:
            tracks_data = split_refined_by_track(df)
            source_kind = "refined_csv"
        return tracks_data, {
            "kind": source_kind,
            "path": str(source_path),
            "columns": sorted(columns),
        }

    raise ValueError(
        f"Could not infer track format from {source_path}. "
        f"Columns were: {sorted(columns)}"
    )


def _node_x_se_values(fit: Dict[str, object], n_nodes: int) -> List[Optional[float]]:
    """
    Expand any available breakpoint-x uncertainty onto the full node list.
    Endpoints currently have no x uncertainty estimate.
    """
    raw = fit.get("breakpoint_se")
    if raw is None:
        return [None] * n_nodes

    values = np.asarray(raw, dtype=float).reshape(-1)
    if values.size == n_nodes:
        return [float(item) if np.isfinite(item) else None for item in values]
    if values.size == max(0, n_nodes - 2):
        return [None] + [float(item) if np.isfinite(item) else None for item in values] + [None]
    return [None] * n_nodes


def _track_quality_payload(
    time_y: np.ndarray,
    fit: Dict[str, object],
) -> Dict[str, object]:
    """
    Recompute one track's fixed-break fit details so exports include consistent
    node values and quality metrics in both shared and independent modes.
    """
    t = np.asarray(time_y[0], dtype=float)
    y = np.asarray(time_y[1], dtype=float)
    breaks = np.asarray(fit["breakpoints"], dtype=float)
    try:
        details = fit_linear_spline_with_breaks_details(time_y, breaks)
        slopes = np.asarray(details["slopes"], dtype=float)
        slope_se = None if details["slope_se"] is None else np.asarray(details["slope_se"], dtype=float)
        break_y = np.asarray(details["break_y"], dtype=float)
        break_y_se = (
            None if details["break_y_se"] is None else np.asarray(details["break_y_se"], dtype=float)
        )
        rss = float(details["rss"])
    except Exception:
        slopes = np.asarray(fit.get("slopes", []), dtype=float)
        slope_se = None
        if fit.get("slope_se") is not None:
            slope_se = np.asarray(fit["slope_se"], dtype=float)
        break_y = np.asarray(fit.get("break_y", np.interp(breaks, t, y)), dtype=float)
        break_y_se = None
        if fit.get("break_y_se") is not None:
            break_y_se = np.asarray(fit["break_y_se"], dtype=float)
        rss = 0.0

    n_obs = int(y.size)
    rmse = float(np.sqrt(rss / n_obs)) if n_obs > 0 else None
    tss = float(np.sum((y - float(np.mean(y))) ** 2)) if n_obs > 0 else 0.0
    if tss > 0:
        r_squared = float(1.0 - rss / tss)
    else:
        r_squared = 1.0 if rss <= 1e-12 else None

    return {
        "breaks": breaks,
        "break_y": break_y,
        "break_y_se": break_y_se,
        "slopes": slopes,
        "slope_se": slope_se,
        "rss": rss,
        "n_obs": n_obs,
        "rmse": rmse,
        "r_squared": r_squared,
        "n_segments": int(len(breaks) - 1),
        "n_params": int(_track_parameter_count(breaks)),
        "segment_counts": _segment_observation_counts(t, breaks),
    }


def _save_spline_fit_artifacts(
    save_path: Union[str, Path],
    tracks_data: Dict[str, np.ndarray],
    fits: Dict[str, Dict[str, object]],
    call_metadata: Dict[str, object],
    source_metadata: Dict[str, object],
    trace: Optional[List[Dict[str, object]]] = None,
) -> Path:
    """
    Save fitted nodes, fit-quality payloads, and metadata into one directory.
    """
    output_dir = Path(save_path).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    nodes_rows: List[Dict[str, object]] = []
    track_quality: Dict[str, Dict[str, object]] = {}
    total_rss = 0.0
    total_obs = 0

    for track_id in sorted(fits):
        fit = fits[track_id]
        quality = _track_quality_payload(tracks_data[track_id], fit)
        breaks = np.asarray(quality["breaks"], dtype=float)
        break_y = np.asarray(quality["break_y"], dtype=float)
        break_y_se = quality["break_y_se"]
        x_se_values = _node_x_se_values(fit, breaks.size)

        for node_index, (x_value, y_value) in enumerate(zip(breaks, break_y)):
            y_se_value = None
            if break_y_se is not None:
                y_se_raw = float(np.asarray(break_y_se, dtype=float)[node_index])
                y_se_value = y_se_raw if np.isfinite(y_se_raw) else None
            nodes_rows.append(
                {
                    "track_id": track_id,
                    "node_index": int(node_index),
                    "x": float(x_value),
                    "x_se": x_se_values[node_index],
                    "y": float(y_value),
                    "y_se": y_se_value,
                    "is_endpoint": bool(node_index == 0 or node_index == breaks.size - 1),
                }
            )

        track_quality[track_id] = {
            "n_obs": int(quality["n_obs"]),
            "n_segments": int(quality["n_segments"]),
            "n_params": int(quality["n_params"]),
            "rss": float(quality["rss"]),
            "rmse": quality["rmse"],
            "r_squared": quality["r_squared"],
            "segment_counts": np.asarray(quality["segment_counts"], dtype=int).tolist(),
            "slopes": np.asarray(quality["slopes"], dtype=float).tolist(),
            "slope_se": None
            if quality["slope_se"] is None
            else np.asarray(quality["slope_se"], dtype=float).tolist(),
            "breakpoints": breaks.tolist(),
            "breakpoint_x_se": x_se_values,
            "break_y": break_y.tolist(),
            "break_y_se": None
            if break_y_se is None
            else np.asarray(break_y_se, dtype=float).tolist(),
        }
        total_rss += float(quality["rss"])
        total_obs += int(quality["n_obs"])

    shared_quality = None
    if fits:
        first_fit = next(iter(fits.values()))
        if "global_breakpoints" in first_fit:
            shared_quality = {
                "global_breakpoints": _jsonify_value(first_fit.get("global_breakpoints")),
                "global_breakpoint_support": _jsonify_value(first_fit.get("global_breakpoint_support")),
                "joint_score": _jsonify_value(first_fit.get("joint_score")),
                "joint_rss": _jsonify_value(first_fit.get("joint_rss")),
                "joint_n_params": _jsonify_value(first_fit.get("joint_n_params")),
                "joint_criterion": _jsonify_value(first_fit.get("joint_criterion")),
            }

    fit_quality_payload = {
        "summary": {
            "n_tracks": int(len(fits)),
            "total_rss": float(total_rss),
            "total_n_obs": int(total_obs),
            "global_rmse": float(np.sqrt(total_rss / total_obs)) if total_obs > 0 else None,
        },
        "shared_fit": shared_quality,
        "tracks": track_quality,
    }

    metadata_payload = {
        "source": _jsonify_value(source_metadata),
        "call": _jsonify_value(call_metadata),
        "artifacts": {
            "nodes_csv": "nodes.csv",
            "fit_quality_json": "fit_quality.json",
            "metadata_json": "metadata.json",
        },
        "trace": _jsonify_value(trace or []),
    }

    pd.DataFrame(nodes_rows).to_csv(output_dir / "nodes.csv", index=False)
    with (output_dir / "fit_quality.json").open("w", encoding="utf-8") as fh:
        json.dump(fit_quality_payload, fh, indent=2)
    with (output_dir / "metadata.json").open("w", encoding="utf-8") as fh:
        json.dump(metadata_payload, fh, indent=2)

    return output_dir


def plot_tracks_from_file(
    file_stem: Optional[str] = None,
    data_folder: Path = DEFAULT_DATA_DIR,
    tracks_data: Optional[Dict[str, np.ndarray]] = None,
    show: bool = True,
) -> Dict[str, np.ndarray]:
    """
    Convenience wrapper: load a file and plot each track's y_px vs time.

    Returns the same dict as tracks_from_file. You can pass preloaded
    ``tracks_data`` (same format) to plot without reloading from disk.
    """
    time_y_data = tracks_data or tracks_from_file(
        file_stem=file_stem, data_folder=data_folder
    )

    plt.figure(figsize=(8, 5))
    for track_id, pair in time_y_data.items():
        time_sec, y_px = pair
        plt.plot(time_sec, y_px, label=track_id)

    plt.xlabel("time_sec")
    plt.ylabel("y_px")
    title_file = file_stem or "selected file"
    plt.title(f"Tracks in {title_file}")
    plt.legend()
    plt.tight_layout()

    if show:
        plt.show()

    return time_y_data


def prune_by_second_derivative_peaks(
    tracks_data: Optional[Dict[str, np.ndarray]] = None,
    file_stem: Optional[str] = None,
    data_folder: Path = DEFAULT_DATA_DIR,
    n_remove: int = 1,
    use_abs: bool = True,
    low_res_bias: float = 0.0,
) -> Dict[str, np.ndarray]:
    """
    Iteratively remove the highest second-derivative points across all tracks.

    Steps repeated until ``n_remove`` total points are removed (not per-track):
      1) For every track with >=3 points, compute d2y/dt2 via np.gradient.
      2) Find the single point (across all tracks) with the highest magnitude
         (or signed) second derivative.
      3) Drop that point from its track, then recompute derivatives on that
         track for the next iteration.
      4) Stop early if no track has >=3 points.

    Args:
        tracks_data: Pre-loaded tracks dict from tracks_from_file. If None, the
            function will load using ``file_stem``/``data_folder``.
        file_stem: Target file stem; used only when tracks_data is None.
        data_folder: Folder containing extracted CSVs (only if loading).
        n_remove: Total number of points to drop across all tracks.
        use_abs: If True, rank by |d2y/dt2|; else by signed d2y/dt2.
        low_res_bias: Exponent that up-weights points sitting in low-resolution
            regions (larger local time spacing). 0.0 disables the bias.

    Returns:
        Dict mapping track_id -> filtered array shaped (2, M) with [time_sec, y_px].
    """
    time_y_data = (
        tracks_data
        if tracks_data is not None
        else tracks_from_file(file_stem=file_stem, data_folder=data_folder)
    )

    # Mutable copies
    t_dict = {k: v[0].copy() for k, v in time_y_data.items()}
    y_dict = {k: v[1].copy() for k, v in time_y_data.items()}
    removed_counts = {k: 0 for k in time_y_data}

    total_removed = 0
    while total_removed < n_remove:
        best_track = None
        best_idx = None
        best_metric = None

        # Evaluate current best point across all eligible tracks
        for track_id in list(t_dict.keys()):
            t = t_dict[track_id]
            y = y_dict[track_id]
            if t.size < 3:
                continue
            t_jittered = t + np.arange(t.size) * 1e-12  # avoid zero dt
            dy_dt = np.gradient(y, t_jittered)
            d2y_dt2 = np.gradient(dy_dt, t_jittered)
            metric_arr = np.abs(d2y_dt2) if use_abs else d2y_dt2

            if low_res_bias != 0.0:
                dt = np.diff(t_jittered)
                # local_dt: average of neighboring spacings, endpoints use nearest spacing
                dt_left = np.empty_like(t_jittered)
                dt_right = np.empty_like(t_jittered)
                dt_left[0] = dt[0]
                dt_left[1:] = dt
                dt_right[:-1] = dt
                dt_right[-1] = dt[-1]
                local_dt = 0.5 * (dt_left + dt_right)
                local_dt = np.maximum(local_dt, 1e-12)  # avoid zeros
                metric_arr = metric_arr * (local_dt ** low_res_bias)

            idx = int(np.argmax(metric_arr))
            metric_val = metric_arr[idx]

            if best_metric is None or metric_val > best_metric:
                best_metric = metric_val
                best_track = track_id
                best_idx = idx

        if best_track is None:
            # No eligible tracks left
            break

        # Remove the selected point
        t_dict[best_track] = np.delete(t_dict[best_track], best_idx)
        y_dict[best_track] = np.delete(y_dict[best_track], best_idx)
        removed_counts[best_track] += 1
        total_removed += 1

        # If a track drops below 3 points, keep it but it won't contribute further

    for track_id, count in removed_counts.items():
        print(f"{track_id}: removed {count} points (total target {n_remove} overall)")

    filtered = {
        track_id: np.vstack((t_dict[track_id], y_dict[track_id]))
        for track_id in t_dict
    }
    return filtered


def piecewise_rates_from_xy(
    time_y: np.ndarray,
    tol: float = 0.1,
    min_points: int = 5,
    max_segments: Optional[int] = None,
) -> List[Tuple[float, float, float]]:
    """
    Derive piecewise-constant slope (dy/dt) regions using up to N linear fits.

    Strategy:
      1) Compute dy/dt.
      2) Search for the widest time interval whose derivative stays within an
         integrated error ``tol`` relative to its mean slope.
      3) Record (t_start, t_end, slope) for that interval, remove it, and repeat
         until no qualifying interval remains or ``max_segments`` is reached.

    Args:
        time_y: np.ndarray shaped (2, N) with time then y.
        tol: Integrated absolute error tolerance in derivative space.
        min_points: Minimum points per segment.
        max_segments: Optional cap on number of segments (None = unlimited).

    Returns:
        List of (t_start, t_end, slope) tuples, ordered by start time.
    """
    if time_y.shape[0] != 2:
        raise ValueError("time_y must have shape (2, N)")
    t = time_y[0]
    y = time_y[1]
    if t.size < min_points:
        return []

    dy_dt = np.gradient(y, t)
    available = np.ones_like(dy_dt, dtype=bool)
    segments: List[Tuple[float, float, float]] = []

    def best_interval(mask: np.ndarray) -> Optional[Tuple[int, int, float, float]]:
        # Returns (start_idx, end_idx, slope, duration) for widest valid interval.
        best = None
        n = mask.size
        i = 0
        while i < n:
            if not mask[i]:
                i += 1
                continue
            j = i
            while j < n and mask[j]:
                j += 1
            block_len = j - i
            if block_len >= min_points:
                block_t = t[i:j]
                block_d = dy_dt[i:j]
                for start in range(block_len - min_points + 1):
                    for end in range(block_len - 1, start + min_points - 2, -1):
                        sub_t = block_t[start : end + 1]
                        sub_d = block_d[start : end + 1]
                        slope = float(np.mean(sub_d))
                        err = np.abs(sub_d - slope)
                        integrated = float(np.trapz(err, sub_t))
                        if integrated <= tol:
                            duration = sub_t[-1] - sub_t[0]
                            if best is None or duration > best[3]:
                                best = (i + start, i + end, slope, duration)
                            break
            i = j
        return best

    while True:
        if max_segments is not None and len(segments) >= max_segments:
            break
        candidate = best_interval(available)
        if candidate is None:
            break
        s_idx, e_idx, slope, _ = candidate
        segments.append((float(t[s_idx]), float(t[e_idx]), slope))
        available[s_idx : e_idx + 1] = False

    segments.sort(key=lambda x: x[0])
    return segments

# `piecewise_rates_from_xy` removed per user request. If you need similar
# functionality later, consider restoring or re-implementing a small
# helper that extracts wide intervals of approximately constant derivative.

# --------------------------------------------------------------------------- #
#  Piecewise linear spline with dynamic knots (using pwlf if available)
# --------------------------------------------------------------------------- #

def fit_linear_spline_auto(
    time_y: np.ndarray,
    max_segments: int = 6,
    criterion: str = "bic",
    min_points: int = 4,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """
    Fit a univariate piecewise-linear spline with data-driven breakpoints.

    This uses the ``pwlf`` library (Piecewise Linear Fit) to fit models with
    k segments (k=1..max_segments), then selects the best k via BIC or AIC.

    Args:
        time_y: np.ndarray shaped (2, N) with time then y.
        max_segments: Maximum number of line segments to consider (>=1).
        criterion: \"bic\" or \"aic\" for model selection.
        min_points: Minimum samples required to attempt fitting.

    Returns:
        (breakpoints, slopes) where:
          - breakpoints: np.ndarray of knot positions in time (len = k+1)
            including the first/last x values.
          - slopes: np.ndarray of length k with slope for each segment.

    Raises:
        ImportError if pwlf is not installed.
        ValueError for invalid inputs.
    """
    if time_y.shape[0] != 2:
        raise ValueError("time_y must have shape (2, N)")
    x = np.asarray(time_y[0], dtype=float)
    y = np.asarray(time_y[1], dtype=float)
    n = x.size
    if n < min_points:
        raise ValueError(f"Need at least {min_points} points, got {n}.")
    if max_segments < 1:
        raise ValueError("max_segments must be >= 1")
    criterion = criterion.lower()
    if criterion not in {"bic", "aic"}:
        raise ValueError("criterion must be 'bic' or 'aic'")

    try:
        import pwlf  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "pwlf is required for fit_linear_spline_auto. "
            "Install with: pip install pwlf"
        ) from exc

    best_score = math.inf
    best_breaks = None
    best_slopes = None
    for k in range(1, max_segments + 1):
        model = pwlf.PiecewiseLinFit(x, y)
        try:
            breaks = model.fit(k)
        except Exception:
            continue  # skip invalid fits
        yhat = model.predict(x)
        rss = float(np.sum((y - yhat) ** 2))
        # Parameter count: k slopes + 1 intercept + (k-1) interior breaks
        p = k + 1 + (k - 1)
        if criterion == "bic":
            score = n * math.log(rss / n) + p * math.log(n)
        else:  # aic
            score = n * math.log(rss / n) + 2 * p
        if score < best_score:
            best_score = score
            best_breaks = np.asarray(breaks)
            # slopes per segment
            best_slopes = np.asarray([model.slopes[i] for i in range(k)])
            try:
                se_all = np.asarray(model.standard_errors())
                # Param order: intercept, slopes (k), breaks (k-1)
                expected_len = 1 + k + (k - 1)
                if se_all.size == expected_len:
                    best_break_se = se_all[-(k - 1):] if k > 1 else np.array([])
                else:
                    best_break_se = None
            except Exception:
                best_break_se = None

    if best_breaks is None or best_slopes is None:
        raise RuntimeError("Failed to fit piecewise linear model.")

    return best_breaks, best_slopes, best_break_se


def fit_linear_spline_with_breaks_details(
    time_y: np.ndarray, breaks: np.ndarray
) -> Dict[str, object]:
    """
    Fit slopes using fixed breakpoints and return a detail dict.

    The returned dict contains:
      - ``slopes``: np.ndarray
      - ``slope_se``: optional np.ndarray
      - ``break_y``: np.ndarray of fitted y values at ``breaks``
      - ``break_y_se``: optional np.ndarray
      - ``rss``: float residual sum of squares
    """
    if time_y.shape[0] != 2:
        raise ValueError("time_y must have shape (2, N)")
    x = np.asarray(time_y[0], dtype=float)
    y = np.asarray(time_y[1], dtype=float)
    try:
        import pwlf  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "pwlf is required for fit_linear_spline_with_breaks. "
            "Install with: pip install pwlf"
        ) from exc
    model = pwlf.PiecewiseLinFit(x, y)
    b_arr = np.asarray(breaks, dtype=float)
    ssr = model.fit_with_breaks(b_arr)
    rss = float(np.asarray(ssr, dtype=float).reshape(-1)[0])
    slopes = np.asarray(model.slopes)
    try:
        # standard_errors includes intercept then slopes; take slopes only.
        se = np.asarray(model.standard_errors())
        if se.size == slopes.size + 1:
            slope_se = se[1:]
        elif se.size == slopes.size:
            slope_se = se
        else:
            slope_se = None
    except Exception:
        slope_se = None

    # Breakpoint values and their uncertainty
    y_at_breaks = model.predict(b_arr)
    try:
        var_pred = np.asarray(model.prediction_variance(b_arr))
        y_se = np.sqrt(var_pred)
    except Exception:
        y_se = None

    return {
        "slopes": slopes,
        "slope_se": slope_se,
        "break_y": y_at_breaks,
        "break_y_se": y_se,
        "rss": rss,
    }


def fit_linear_spline_with_breaks(
    time_y: np.ndarray, breaks: np.ndarray
) -> Tuple[np.ndarray, Optional[np.ndarray], np.ndarray, Optional[np.ndarray]]:
    """
    Fit slopes using fixed breakpoints and return slopes, slope SE,
    y at breakpoints, and y SE (if available).
    """
    details = fit_linear_spline_with_breaks_details(time_y, breaks)
    return (
        np.asarray(details["slopes"]),
        details["slope_se"],
        np.asarray(details["break_y"]),
        details["break_y_se"],
    )


def _global_time_bounds(tracks_data: Dict[str, np.ndarray]) -> Tuple[float, float]:
    all_t = np.concatenate([arr[0] for arr in tracks_data.values()])
    return float(np.min(all_t)), float(np.max(all_t))


def _track_breaks_from_global_breaks(time_y: np.ndarray, global_breaks: np.ndarray) -> np.ndarray:
    """
    Clip a shared breakpoint grid down to the portion covered by a single track.
    """
    t = np.asarray(time_y[0], dtype=float)
    tmin = float(np.min(t))
    tmax = float(np.max(t))
    b_arr = np.asarray(global_breaks, dtype=float)
    inner = b_arr[(b_arr > tmin) & (b_arr < tmax)]
    track_breaks = np.concatenate(([tmin], inner, [tmax]))
    return np.unique(track_breaks)


def _segment_observation_counts(t: np.ndarray, breaks: np.ndarray) -> np.ndarray:
    """
    Count observations per segment, assigning shared endpoints to the segment on the right.
    """
    counts = np.zeros(len(breaks) - 1, dtype=int)
    for i in range(len(counts)):
        left = breaks[i]
        right = breaks[i + 1]
        if i == len(counts) - 1:
            mask = (t >= left) & (t <= right)
        else:
            mask = (t >= left) & (t < right)
        counts[i] = int(np.count_nonzero(mask))
    return counts


def _support_window_from_times(
    times: np.ndarray,
    support_window_factor: Optional[float],
) -> Optional[float]:
    """
    Convert a typical sampling interval into a breakpoint-support window.
    """
    if support_window_factor is None:
        return None
    factor = float(support_window_factor)
    if not np.isfinite(factor) or factor <= 0:
        return None

    t_sorted = np.unique(np.sort(np.asarray(times, dtype=float)))
    if t_sorted.size < 2:
        return factor * 1e-12

    diffs = np.diff(t_sorted)
    positive_diffs = diffs[diffs > 0]
    if positive_diffs.size:
        base_scale = float(np.median(positive_diffs))
    else:
        span = float(t_sorted[-1] - t_sorted[0])
        base_scale = span if span > 0 else 1e-12
    return factor * base_scale


def _breakpoint_support_record(
    times: np.ndarray,
    tau: float,
    support_window_factor: Optional[float],
) -> Optional[Dict[str, float]]:
    """
    Return left/right gap diagnostics for one breakpoint, or None if support
    checks are disabled.
    """
    window = _support_window_from_times(times, support_window_factor)
    if window is None:
        return None

    t_sorted = np.unique(np.sort(np.asarray(times, dtype=float)))
    left_idx = int(np.searchsorted(t_sorted, tau, side="left")) - 1
    right_idx = int(np.searchsorted(t_sorted, tau, side="right"))

    left_gap = math.inf if left_idx < 0 else float(tau - t_sorted[left_idx])
    right_gap = math.inf if right_idx >= t_sorted.size else float(t_sorted[right_idx] - tau)
    gap_width = float(left_gap + right_gap)
    supported = (
        np.isfinite(left_gap)
        and np.isfinite(right_gap)
        and gap_width <= 2.0 * float(window)
    )
    return {
        "breakpoint": float(tau),
        "left_gap": float(left_gap),
        "right_gap": float(right_gap),
        "gap_width": gap_width,
        "window": float(window),
        "supported": bool(supported),
    }


def _pooled_active_times_for_breakpoint(
    tracks_data: Dict[str, np.ndarray],
    tau: float,
) -> Tuple[np.ndarray, List[str]]:
    """
    Pool all observation times from tracks whose time span contains ``tau``.
    """
    pooled_times = []
    active_track_ids: List[str] = []
    for track_id, arr in tracks_data.items():
        t = np.asarray(arr[0], dtype=float)
        if float(np.min(t)) < tau < float(np.max(t)):
            pooled_times.append(t)
            active_track_ids.append(track_id)
    if not pooled_times:
        return np.array([], dtype=float), []
    return np.unique(np.concatenate(pooled_times)), active_track_ids


def _unsupported_global_breakpoints(
    tracks_data: Dict[str, np.ndarray],
    global_breaks: np.ndarray,
    support_window_factor: Optional[float],
) -> List[Dict[str, float]]:
    """
    Return interior shared breakpoints that fall inside overly wide pooled
    observation gaps.
    """
    unsupported: List[Dict[str, float]] = []
    for tau in np.asarray(global_breaks, dtype=float)[1:-1]:
        pooled_times, active_track_ids = _pooled_active_times_for_breakpoint(
            tracks_data,
            float(tau),
        )
        if not active_track_ids:
            continue
        record = _breakpoint_support_record(pooled_times, float(tau), support_window_factor)
        if record is not None and not bool(record["supported"]):
            record["active_track_ids"] = active_track_ids
            record["n_active_tracks"] = len(active_track_ids)
            unsupported.append(record)
    return unsupported


def _track_parameter_count(track_breaks: np.ndarray) -> int:
    """
    Continuous piecewise-linear track fit parameter count:
      1 intercept + 1 base slope + one slope change per interior break.
    """
    return len(track_breaks)


def _fit_track_on_shared_breaks(
    time_y: np.ndarray,
    global_breaks: np.ndarray,
    min_points: int,
    points_per_segment: int = 50,
) -> Optional[Dict[str, object]]:
    """
    Fit one track using only the subset of shared breakpoints inside its domain.
    """
    if time_y.shape[0] != 2:
        raise ValueError("time_y must have shape (2, N)")
    t = np.asarray(time_y[0], dtype=float)
    if t.size < min_points:
        return None

    track_breaks = _track_breaks_from_global_breaks(time_y, global_breaks)
    counts = _segment_observation_counts(t, track_breaks)
    if counts.size == 0:
        return None

    try:
        details = fit_linear_spline_with_breaks_details(time_y, track_breaks)
    except Exception:
        return None
    rss = float(details["rss"])
    if not np.isfinite(rss):
        return None

    slopes = np.asarray(details["slopes"], dtype=float)
    break_y = np.asarray(details["break_y"], dtype=float)
    spline_xy = sample_piecewise_linear_curve(
        track_breaks,
        slopes,
        break_y=break_y,
        points_per_segment=points_per_segment,
    )
    return {
        "track_breaks": track_breaks,
        "segment_counts": counts,
        "slopes": slopes,
        "slope_se": details["slope_se"],
        "break_y": break_y,
        "break_y_se": details["break_y_se"],
        "spline_xy": spline_xy,
        "rss": rss,
        "n_segments": len(track_breaks) - 1,
        "n_params": _track_parameter_count(track_breaks),
    }


def _diagnose_track_shared_break_failure(
    track_id: str,
    time_y: np.ndarray,
    global_breaks: np.ndarray,
    min_points: int,
) -> Optional[Dict[str, object]]:
    """
    Return a compact failure record for a track/shared-break fit, or None if valid.
    """
    if time_y.shape[0] != 2:
        return {
            "track_id": track_id,
            "reason": "invalid_shape",
            "shape": tuple(int(v) for v in time_y.shape),
        }

    t = np.asarray(time_y[0], dtype=float)
    if t.size < min_points:
        return {
            "track_id": track_id,
            "reason": "too_few_points",
            "n_points": int(t.size),
            "min_points": int(min_points),
        }

    track_breaks = _track_breaks_from_global_breaks(time_y, global_breaks)
    counts = _segment_observation_counts(t, track_breaks)
    rounded_breaks = np.round(track_breaks, 6).tolist()
    count_list = counts.astype(int).tolist()

    if counts.size == 0:
        return {
            "track_id": track_id,
            "reason": "no_segments",
            "track_breaks": rounded_breaks,
        }

    try:
        details = fit_linear_spline_with_breaks_details(time_y, track_breaks)
    except Exception as exc:
        return {
            "track_id": track_id,
            "reason": "fit_exception",
            "track_breaks": rounded_breaks,
            "segment_counts": count_list,
            "exception": repr(exc),
        }

    rss = float(details["rss"])
    if not np.isfinite(rss):
        return {
            "track_id": track_id,
            "reason": "nonfinite_rss",
            "track_breaks": rounded_breaks,
            "segment_counts": count_list,
            "rss": rss,
        }

    return None


def _candidate_break_grid(
    tracks_data: Dict[str, np.ndarray],
    fitted_breaks: List[np.ndarray],
) -> np.ndarray:
    """
    Candidate shared break locations from observed timestamps plus first-pass proposals.
    """
    # Use only pooled interior proposals from the first-pass fits as the
    # candidate grid. This avoids including raw observed timestamps (which
    # are often determined by acquisition timing and can dominate the
    # candidate set), and focuses the joint search on breakpoints proposed
    # by the per-track exploratory fits.
    global_min, global_max = _global_time_bounds(tracks_data)
    interior = [bk[1:-1] for bk in fitted_breaks if len(bk) > 2]
    if not interior:
        return np.array([])
    pooled_interior = np.concatenate(interior)
    pooled_interior = pooled_interior[(pooled_interior > global_min) & (pooled_interior < global_max)]
    if pooled_interior.size == 0:
        return np.array([])
    return np.unique(pooled_interior)

def _information_criterion_score(
    rss: float,
    n_obs: int,
    n_params: int,
    criterion: str,
) -> float:
    rss = max(float(rss), 1e-12)
    if criterion == "bic":
        return n_obs * math.log(rss / n_obs) + n_params * math.log(n_obs)
    return n_obs * math.log(rss / n_obs) + 2 * n_params


def _seed_debug_summary(
    seed: np.ndarray,
    standard_breaks: np.ndarray = _SEED_DEBUG_STANDARD_BREAKS,
    margin: float = _SEED_DEBUG_MATCH_MARGIN,
) -> Dict[str, object]:
    """
    Summarize how a seed aligns with the current nine-breakpoint reference.
    """
    seed_values = np.sort(np.asarray(seed, dtype=float))
    standard_values = np.sort(np.asarray(standard_breaks, dtype=float))

    matched_standard = np.zeros(standard_values.size, dtype=bool)
    match_count = 0
    duplicate_match_count = 0

    for seed_value in seed_values:
        candidate_idx = np.flatnonzero(np.abs(standard_values - seed_value) <= margin)
        if candidate_idx.size == 0:
            continue

        unmatched_idx = [idx for idx in candidate_idx if not matched_standard[idx]]
        if unmatched_idx:
            best_idx = min(
                unmatched_idx,
                key=lambda idx: (abs(float(standard_values[idx]) - float(seed_value)), idx),
            )
            matched_standard[best_idx] = True
            match_count += 1
        else:
            duplicate_match_count += 1

    close_seed_pairs: List[Tuple[float, float]] = []
    if seed_values.size >= 2:
        overlap_threshold = 2.0 * margin
        for left, right in zip(seed_values[:-1], seed_values[1:]):
            if float(right) - float(left) <= overlap_threshold:
                close_seed_pairs.append((round(float(left), 6), round(float(right), 6)))

    return {
        "match_count": int(match_count),
        "duplicate_match_count": int(duplicate_match_count),
        "close_seed_pairs": close_seed_pairs,
    }


def _format_seed_debug_summary(
    seed: np.ndarray,
    margin: float = _SEED_DEBUG_MATCH_MARGIN,
) -> str:
    summary = _seed_debug_summary(seed, margin=margin)
    return (
        f"matches={summary['match_count']} "
        f"duplicate_matches={summary['duplicate_match_count']} "
        f"close_seed_pairs={summary['close_seed_pairs']}"
    )


def _format_seed_invalid_diagnostic(diagnostic: Optional[Dict[str, object]]) -> str:
    """
    Render a compact one-line reason for a rejected shared-break seed.
    """
    if not diagnostic:
        return "reason=unknown"

    track_id = diagnostic.get("track_id", "?")
    reason = diagnostic.get("reason", "unknown")

    if reason == "weak_global_support":
        unsupported_breakpoints = diagnostic.get("unsupported_breakpoints", [])
        formatted_breaks = [
            f"{float(item['breakpoint']):.6f}"
            f"(left={float(item['left_gap']):.6f},"
            f" right={float(item['right_gap']):.6f},"
            f" width={float(item['gap_width']):.6f},"
            f" window={float(item['window']):.6f},"
            f" active_tracks={int(item['n_active_tracks'])})"
            for item in unsupported_breakpoints
        ]
        return f"reason={reason} unsupported_breakpoints={formatted_breaks}"

    if reason == "too_few_points":
        return (
            f"track={track_id} reason={reason} "
            f"n_points={diagnostic.get('n_points')} "
            f"min_points={diagnostic.get('min_points')}"
        )

    if reason == "fit_exception":
        return (
            f"track={track_id} reason={reason} "
            f"exception={diagnostic.get('exception')}"
        )

    if reason == "nonfinite_rss":
        return (
            f"track={track_id} reason={reason} "
            f"rss={diagnostic.get('rss')}"
        )

    if reason == "invalid_shape":
        return (
            f"track={track_id} reason={reason} "
            f"shape={diagnostic.get('shape')}"
        )

    if reason == "no_segments":
        return f"track={track_id} reason={reason}"

    return f"track={track_id} reason={reason}"

def _group_debug_hypothesis_seeds(
    hypothesis_seeds: Optional[object],
    global_min: float,
    global_max: float,
    debug: bool = False,
) -> Dict[int, List[np.ndarray]]:
    """
    Normalize optional manual interior-breakpoint seeds and bucket them by
    target segment count.
    """
    if hypothesis_seeds is None:
        return {}

    if isinstance(hypothesis_seeds, np.ndarray):
        raw_seeds = (hypothesis_seeds,)
    elif isinstance(hypothesis_seeds, (list, tuple)):
        if len(hypothesis_seeds) == 0:
            raw_seeds = ()
        elif np.isscalar(hypothesis_seeds[0]):
            raw_seeds = (hypothesis_seeds,)
        else:
            raw_seeds = tuple(hypothesis_seeds)
    else:
        raw_seeds = (hypothesis_seeds,)

    grouped: Dict[int, List[np.ndarray]] = {}
    seen = set()
    for seed_idx, seed in enumerate(raw_seeds, start=1):
        try:
            values = np.asarray(seed, dtype=float).reshape(-1)
        except Exception as exc:
            if debug:
                print(
                    f"[seed-debug] skipped hypothesis seed {seed_idx}: "
                    f"could not coerce to floats ({exc!r})"
                )
            continue
        if not np.all(np.isfinite(values)):
            if debug:
                print(
                    f"[seed-debug] skipped hypothesis seed {seed_idx}: "
                    "contains non-finite values"
                )
            continue

        values = np.sort(values)
        if np.any(values <= global_min) or np.any(values >= global_max):
            if debug:
                print(
                    f"[seed-debug] skipped hypothesis seed {seed_idx}: "
                    f"must lie strictly inside ({global_min:.6f}, {global_max:.6f})"
                )
            continue

        key = tuple(np.round(values, 12))
        if key in seen:
            continue
        seen.add(key)
        grouped.setdefault(int(values.size) + 1, []).append(values)

    return grouped

def _shared_break_seed_sets(
    n_segments: int,
    candidate_grid: np.ndarray,
    bandwidth: Optional[float] = None,
    max_seed_sets: int = 16,
    debug: bool = False,
) -> List[np.ndarray]:
    """
    Generate MeanShift-based starting breakpoint configurations from the
    unique global candidate breakpoint pool.
    """
    n_breaks = n_segments - 1
    if n_breaks == 0:
        return [np.array([], dtype=float)]
    if candidate_grid.size < n_breaks:
        return []

    grid = np.sort(np.asarray(candidate_grid, dtype=float))
    samples = grid.reshape(-1, 1)
    if bandwidth is None:
        bandwidth = estimate_bandwidth(samples, quantile=0.2, n_samples=min(len(samples), 500))
    if bandwidth is None or not np.isfinite(bandwidth) or bandwidth <= 0:
        bandwidth = 1e-3

    ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
    ms.fit(samples)

    cluster_centers = np.asarray(ms.cluster_centers_, dtype=float).reshape(-1)
    labels = np.asarray(ms.labels_, dtype=int)
    unique_labels, counts = np.unique(labels, return_counts=True)
    sorted_indices = np.argsort(counts)[::-1]

    if debug:
        raw_pairs = [
            (float(cluster_centers[unique_labels[idx]]), int(counts[idx]))
            for idx in sorted_indices[:10]
        ]
        print(
            f"[seed-debug] k={n_segments}: candidate_grid_size={grid.size}, "
            f"bandwidth={float(bandwidth):.6f}, raw_meanshift_centers={raw_pairs}"
        )

    # Project cluster centers back onto the actual admissible breakpoint grid,
    # then retain one support count per unique projected location.
    support_by_break: Dict[float, int] = {}
    for idx in sorted_indices:
        center = float(cluster_centers[unique_labels[idx]])
        projected = float(grid[int(np.argmin(np.abs(grid - center)))])
        support = int(counts[idx])
        previous = support_by_break.get(projected)
        if previous is None or support > previous:
            support_by_break[projected] = support

    ranked_breaks = [
        (breakpoint, support_by_break[breakpoint])
        for breakpoint in sorted(
            support_by_break,
            key=lambda bp: (-support_by_break[bp], bp),
        )
    ]
    if debug:
        print(
            f"[seed-debug] k={n_segments}: projected_ranked_breaks="
            f"{[(round(bp, 6), support) for bp, support in ranked_breaks[:12]]}"
        )
    if len(ranked_breaks) < n_breaks:
        return []

    top_pool_size = min(len(ranked_breaks), max(n_breaks, 10))
    top_breaks = ranked_breaks[:top_pool_size]
    scored_seed_candidates = []
    for combo in itertools.combinations(top_breaks, n_breaks):
        values = np.sort(np.asarray([item[0] for item in combo], dtype=float))
        score = int(sum(item[1] for item in combo))
        scored_seed_candidates.append((score, values))

    scored_seed_candidates.sort(
        key=lambda item: (
            -item[0],
            tuple(np.round(item[1], 12)),
        )
    )

    seeds: List[np.ndarray] = []
    seen = set()
    for _, values in scored_seed_candidates:
        key = tuple(np.round(values, 12))
        if key in seen:
            continue
        seen.add(key)
        seeds.append(values)
        if len(seeds) >= max_seed_sets:
            break

    if debug:
        preview = [_format_seed_debug_summary(seed) for seed in seeds[:8]]
        print(
            f"[seed-debug] k={n_segments}: generated {len(seeds)} seed set(s); "
            f"preview={preview}"
        )

    return seeds


def optimize_seeds(
    n_segments,
    candidate_grid,
    evaluate,
    max_passes,
    global_min,
    global_max,
    bandwidth=None,
    debug: bool = False,
    diagnose=None,
    manual_seeds: Optional[List[np.ndarray]] = None,
    validate_manual_seeds: bool = True,
    trace: Optional[List[Dict[str, object]]] = None,
):
    """
    Optimize MeanShift-derived breakpoint seeds by coordinate descent.
    """
    n_breaks = n_segments - 1
    best: Optional[Dict[str, object]] = None

    seed_results: List[Tuple[np.ndarray, Optional[Dict[str, object]]]] = []
    auto_seeds = _shared_break_seed_sets(
        n_segments,
        candidate_grid,
        bandwidth=bandwidth,
        debug=debug,
    )
    manual_seed_list = [np.sort(np.asarray(seed, dtype=float)) for seed in (manual_seeds or [])]
    manual_seed_keys = {tuple(np.round(seed, 12)) for seed in manual_seed_list}

    initial_seeds: List[np.ndarray] = []
    seen = set()
    for seed in manual_seed_list + auto_seeds:
        values = np.sort(np.asarray(seed, dtype=float))
        key = tuple(np.round(values, 12))
        if key in seen:
            continue
        seen.add(key)
        initial_seeds.append(values)

    if debug and manual_seed_list:
        preview = [np.round(seed, 6).tolist() for seed in manual_seed_list[:8]]
        print(
            f"[seed-debug] k={n_segments}: injected {len(manual_seed_list)} "
            f"hypothesis seed(s); preview={preview}"
        )
    if manual_seed_list:
        _append_fit_trace_event(
            trace,
            "manual_seeds_injected",
            k=n_segments,
            seeds=[np.round(seed, 6).tolist() for seed in manual_seed_list],
        )

    for seed_idx, seed in enumerate(initial_seeds, start=1):
        current = seed.copy()
        seed_key = tuple(np.round(current, 12))
        is_manual = seed_key in manual_seed_keys
        seed_label = "hypothesis seed" if is_manual else "seed"
        current_result = evaluate(current)
        if current_result is None:
            diagnostic_text = ""
            if diagnose is not None:
                diagnostic = diagnose(current)
                if diagnostic is not None:
                    diagnostic_text = " " + _format_seed_invalid_diagnostic(diagnostic)
            if is_manual and not validate_manual_seeds:
                if debug:
                    print(
                        f"[seed-debug] k={n_segments}: {seed_label} {seed_idx} "
                        f"bypassed initial validity check "
                        f"values={np.round(seed, 6).tolist()}{diagnostic_text}"
                    )
                _append_fit_trace_event(
                    trace,
                    "seed_bypassed_initial_validity",
                    k=n_segments,
                    seed_index=seed_idx,
                    manual=is_manual,
                    seed=np.round(seed, 6).tolist(),
                    diagnostic=diagnostic_text.strip() or None,
                )
                seed_results.append((current, None))
                continue
            if debug:
                print(
                    f"[seed-debug] k={n_segments}: {seed_label} {seed_idx} invalid "
                    f"{_format_seed_debug_summary(seed)} "
                    f"values={np.round(seed, 6).tolist()}{diagnostic_text}"
                )
            _append_fit_trace_event(
                trace,
                "seed_invalid",
                k=n_segments,
                seed_index=seed_idx,
                manual=is_manual,
                seed=np.round(seed, 6).tolist(),
                diagnostic=diagnostic_text.strip() or None,
            )
            continue
        if debug:
            print(
                f"[seed-debug] k={n_segments}: {seed_label} {seed_idx} valid "
                f"{_format_seed_debug_summary(seed)} "
                f"values={np.round(seed, 6).tolist()} "
                f"score={float(current_result['score']):.6f}"
            )
        _append_fit_trace_event(
            trace,
            "seed_valid",
            k=n_segments,
            seed_index=seed_idx,
            manual=is_manual,
            seed=np.round(seed, 6).tolist(),
            score=float(current_result["score"]),
        )
        seed_results.append((current, current_result))

    if debug and not seed_results:
        print(f"[seed-debug] k={n_segments}: no valid seed sets survived initial evaluation")
    if not seed_results:
        _append_fit_trace_event(trace, "no_valid_initial_seeds", k=n_segments)

    for current, current_result in seed_results:

        improved = True
        passes = 0
        
        while improved and passes < max_passes:
            improved = False
            passes += 1
            
            # EDGE CASE: Single breakpoint (Pairwise is impossible)
            if n_breaks == 1:
                lower = global_min
                upper = global_max
                feasible = candidate_grid[(candidate_grid > lower) & (candidate_grid < upper)]
                if feasible.size == 0:
                    continue

                local_best = current_result
                for cand in feasible:
                    trial = current.copy()
                    trial[0] = cand
                    trial_result = evaluate(trial)
                    if trial_result is None:
                        continue
                    if local_best is None or trial_result["score"] < local_best["score"] - 1e-9:
                        local_best = trial_result

                if local_best is not current_result:
                    old_score = None if current_result is None else float(current_result["score"])
                    if debug:
                        old_summary = _format_seed_debug_summary(current)
                    current_result = local_best
                    if current_result is None:
                        continue
                    current = np.asarray(current_result["interior_breaks"], dtype=float)
                    if debug:
                        if old_score is None:
                            print(
                                f"[seed-debug] k={n_segments}: single-break recovery "
                                f"pass={passes} {old_summary} -> "
                                f"{_format_seed_debug_summary(current)} "
                                f"score={float(current_result['score']):.6f}"
                            )
                        else:
                            print(
                                f"[seed-debug] k={n_segments}: single-break improvement "
                                f"pass={passes} {old_summary} -> "
                                f"{_format_seed_debug_summary(current)} "
                                f"score {old_score:.6f} -> {float(current_result['score']):.6f}"
                            )
                    _append_fit_trace_event(
                        trace,
                        "single_break_update",
                        k=n_segments,
                        pass_index=passes,
                        previous_score=old_score,
                        new_score=float(current_result["score"]),
                        new_seed=np.round(current, 6).tolist(),
                    )
                    improved = True
                continue # Skip pairwise logic
            
            # PRIMARY STRATEGY: Pairwise Coordinate Descent
            for idx in range(n_breaks - 1):
                # Bounds for the PAIR of breakpoints: current[idx] and current[idx+1]
                lower = global_min if idx == 0 else current[idx - 1]
                upper = global_max if idx + 1 == n_breaks - 1 else current[idx + 2]
                
                feasible = candidate_grid[(candidate_grid > lower) & (candidate_grid < upper)]
                if feasible.size < 2:
                    continue

                local_best = current_result
                
                # Evaluate all possible pairs within the feasible window
                for cand1, cand2 in itertools.combinations(feasible, 2):
                    trial = current.copy()
                    trial[idx] = cand1
                    trial[idx + 1] = cand2
                    # sorting is implicitly handled by itertools.combinations
                    
                    trial_result = evaluate(trial)
                    if trial_result is None:
                        continue
                    if local_best is None or trial_result["score"] < local_best["score"] - 1e-9:
                        local_best = trial_result

                if local_best is not current_result:
                    old_score = None if current_result is None else float(current_result["score"])
                    if debug:
                        old_summary = _format_seed_debug_summary(current)
                    current_result = local_best
                    if current_result is None:
                        continue
                    current = np.asarray(current_result["interior_breaks"], dtype=float)
                    if debug:
                        if old_score is None:
                            print(
                                f"[seed-debug] k={n_segments}: pair recovery idx={idx} "
                                f"pass={passes} {old_summary} -> "
                                f"{_format_seed_debug_summary(current)} "
                                f"score={float(current_result['score']):.6f}"
                            )
                        else:
                            print(
                                f"[seed-debug] k={n_segments}: pair improvement idx={idx} "
                                f"pass={passes} {old_summary} -> "
                                f"{_format_seed_debug_summary(current)} "
                                f"score {old_score:.6f} -> {float(current_result['score']):.6f}"
                            )
                    _append_fit_trace_event(
                        trace,
                        "pair_update",
                        k=n_segments,
                        pair_index=idx,
                        pass_index=passes,
                        previous_score=old_score,
                        new_score=float(current_result["score"]),
                        new_seed=np.round(current, 6).tolist(),
                    )
                    improved = True

        if current_result is not None and (best is None or current_result["score"] < best["score"]):
            best = current_result

    if debug and best is not None:
        print(
            f"[seed-debug] k={n_segments}: best optimized seed="
            f"{_format_seed_debug_summary(np.asarray(best['interior_breaks'], dtype=float))} "
            f"score={float(best['score']):.6f}"
        )
    if best is not None:
        _append_fit_trace_event(
            trace,
            "best_seed_selected",
            k=n_segments,
            seed=np.round(np.asarray(best["interior_breaks"], dtype=float), 6).tolist(),
            score=float(best["score"]),
        )

    return best


def _evaluate_shared_break_configuration(
    tracks_data: Dict[str, np.ndarray],
    global_breaks: np.ndarray,
    criterion: str,
    min_points: int,
    support_window_factor: Optional[float] = 2.0,
) -> Optional[Dict[str, object]]:
    """
    Score a shared breakpoint configuration by jointly refitting all active tracks.
    """
    if support_window_factor is not None:
        unsupported_breaks = _unsupported_global_breakpoints(
            tracks_data,
            global_breaks,
            support_window_factor,
        )
        if unsupported_breaks:
            return None

    total_obs = int(sum(arr.shape[1] for arr in tracks_data.values()))
    total_rss = 0.0
    total_params = int(len(global_breaks) - 2)  # shared interior break locations
    track_fits: Dict[str, Dict[str, object]] = {}

    for track_id, arr in tracks_data.items():
        fit = _fit_track_on_shared_breaks(
            arr,
            global_breaks,
            min_points=min_points,
        )
        if fit is None:
            return None
        track_fits[track_id] = fit
        total_rss += float(fit["rss"])
        total_params += int(fit["n_params"])

    score = _information_criterion_score(total_rss, total_obs, total_params, criterion)
    return {
        "global_breaks": np.asarray(global_breaks, dtype=float),
        "track_fits": track_fits,
        "rss": float(total_rss),
        "score": float(score),
        "n_params": total_params,
        "criterion": criterion,
    }


def _optimize_shared_breaks_for_segment_count(
    tracks_data: Dict[str, np.ndarray],
    n_segments: int,
    candidate_grid: np.ndarray,
    criterion: str,
    min_points: int,
    support_window_factor: Optional[float] = 2.0,
    max_passes: int = 12,
    bandwidth: Optional[float] = None,
    debug: bool = False,
    manual_seeds: Optional[List[np.ndarray]] = None,
    validate_manual_seeds: bool = True,
    trace: Optional[List[Dict[str, object]]] = None,
) -> Optional[Dict[str, object]]:
    """
    Coordinate-descent refinement of a shared breakpoint set for a fixed segment count.
    """
    n_breaks = n_segments - 1
    global_min, global_max = _global_time_bounds(tracks_data)
    manual_seed_list = list(manual_seeds or [])
    if n_breaks > candidate_grid.size and not manual_seed_list:
        return None

    cache: Dict[Tuple[float, ...], Optional[Dict[str, object]]] = {}
    diagnostic_cache: Dict[Tuple[float, ...], Optional[Dict[str, object]]] = {}

    def evaluate(interior_breaks: np.ndarray) -> Optional[Dict[str, object]]:
        interior = np.sort(np.asarray(interior_breaks, dtype=float))
        key = tuple(np.round(interior, 12))
        if key in cache:
            return cache[key]
        global_breaks = np.concatenate(([global_min], interior, [global_max]))
        result = _evaluate_shared_break_configuration(
            tracks_data,
            global_breaks,
            criterion=criterion,
            min_points=min_points,
            support_window_factor=support_window_factor,
        )
        if result is not None:
            result["interior_breaks"] = interior
        cache[key] = result
        return result

    def diagnose(interior_breaks: np.ndarray) -> Optional[Dict[str, object]]:
        interior = np.sort(np.asarray(interior_breaks, dtype=float))
        key = tuple(np.round(interior, 12))
        if key in diagnostic_cache:
            return diagnostic_cache[key]
        if key in cache and cache[key] is not None:
            diagnostic_cache[key] = None
            return None

        global_breaks = np.concatenate(([global_min], interior, [global_max]))
        if support_window_factor is not None:
            unsupported_breaks = _unsupported_global_breakpoints(
                tracks_data,
                global_breaks,
                support_window_factor,
            )
            if unsupported_breaks:
                diagnostic = {
                    "reason": "weak_global_support",
                    "unsupported_breakpoints": [
                        {
                            "breakpoint": round(float(item["breakpoint"]), 6),
                            "left_gap": round(float(item["left_gap"]), 6),
                            "right_gap": round(float(item["right_gap"]), 6),
                            "gap_width": round(float(item["gap_width"]), 6),
                            "window": round(float(item["window"]), 6),
                            "n_active_tracks": int(item["n_active_tracks"]),
                            "active_track_ids": list(item["active_track_ids"]),
                        }
                        for item in unsupported_breaks
                    ],
                }
                diagnostic_cache[key] = diagnostic
                return diagnostic

        first_failure: Optional[Dict[str, object]] = None
        for track_id, arr in tracks_data.items():
            failure = _diagnose_track_shared_break_failure(
                track_id,
                arr,
                global_breaks,
                min_points=min_points,
            )
            if failure is not None:
                first_failure = failure
                break
        diagnostic_cache[key] = first_failure
        return first_failure

    best = optimize_seeds(
        n_segments=n_segments,
        candidate_grid=candidate_grid,
        evaluate=evaluate,
        diagnose=diagnose,
        max_passes=max_passes,
        global_min=global_min,
        global_max=global_max,
        bandwidth=bandwidth,
        debug=debug,
        manual_seeds=manual_seed_list,
        validate_manual_seeds=validate_manual_seeds,
        trace=trace,
    )
    return best

def _shared_break_support(
    tracks_data: Dict[str, np.ndarray],
    global_breaks: np.ndarray,
) -> np.ndarray:
    """
    Number of tracks that actively span each shared interior break.
    """
    interior = np.asarray(global_breaks, dtype=float)[1:-1]
    support = []
    for tau in interior:
        count = 0
        for arr in tracks_data.values():
            t = np.asarray(arr[0], dtype=float)
            if not (float(np.min(t)) < tau < float(np.max(t))):
                continue
            count += 1
        support.append(count)
    return np.asarray(support, dtype=int)


def fit_shared_breakpoints_joint(
    tracks_data: Dict[str, np.ndarray],
    fitted_breaks: List[np.ndarray],
    max_segments: int = 6,
    criterion: str = "bic",
    min_points: int = 4,
    show_progress: bool = False,
    breakpoint_cluster_bandwidth: Optional[float] = None,
    support_window_factor: Optional[float] = 2.0,
    debug_hypothesis_seeds: Optional[Tuple[np.ndarray, ...]] = None,
    debug_validate_hypothesis_seeds: bool = True,
    trace: Optional[List[Dict[str, object]]] = None,
) -> Dict[str, object]:
    """
    Jointly estimate one shared breakpoint set, then refit each track on its active subset.

    Shared breakpoints are global in time, but each track only activates the subset
    that lies inside its observed time span. Global seed validation uses pooled
    observation times across all active tracks, while per-track validity is
    determined only by whether the fixed-break refit succeeds with finite RSS.
    """
    global_min, global_max = _global_time_bounds(tracks_data)
    candidate_grid = _candidate_break_grid(tracks_data, fitted_breaks)
    manual_seed_map = _group_debug_hypothesis_seeds(
        debug_hypothesis_seeds,
        global_min=global_min,
        global_max=global_max,
        debug=show_progress,
    )
    best_overall: Optional[Dict[str, object]] = None
    start_time = time.perf_counter()
    _append_fit_trace_event(
        trace,
        "shared_fit_start",
        n_tracks=len(tracks_data),
        max_segments=max_segments,
        candidate_grid=candidate_grid,
        criterion=criterion,
    )

    if show_progress:
        print(
            "[shared-fit] starting joint breakpoint search "
            f"for {len(tracks_data)} tracks, up to {max_segments} segments, "
            f"{candidate_grid.size} candidate break times"
        )
        if candidate_grid.size:
            preview_head = np.round(candidate_grid[: min(8, candidate_grid.size)], 6).tolist()
            preview_tail = np.round(candidate_grid[-min(8, candidate_grid.size):], 6).tolist()
            print(
                f"[seed-debug] candidate_grid preview head={preview_head} tail={preview_tail}"
            )
        if manual_seed_map:
            manual_summary = {
                k: [np.round(seed, 6).tolist() for seed in seeds]
                for k, seeds in sorted(manual_seed_map.items())
            }
            print(f"[seed-debug] manual hypothesis seeds by k={manual_summary}")
    if manual_seed_map:
        _append_fit_trace_event(
            trace,
            "manual_seed_map",
            seeds_by_k={
                k: [np.round(seed, 6).tolist() for seed in seeds]
                for k, seeds in sorted(manual_seed_map.items())
            },
        )

    for n_segments in range(1, max_segments + 1):
        iter_start = time.perf_counter()
        _append_fit_trace_event(trace, "segment_count_start", k=n_segments)
        if show_progress:
            print(f"[shared-fit] evaluating {n_segments} shared segment(s)")
        best_for_k = _optimize_shared_breaks_for_segment_count(
            tracks_data,
            n_segments=n_segments,
            candidate_grid=candidate_grid,
            criterion=criterion,
            min_points=min_points,
            support_window_factor=support_window_factor,
            bandwidth=breakpoint_cluster_bandwidth,
            debug=show_progress,
            manual_seeds=manual_seed_map.get(n_segments),
            validate_manual_seeds=debug_validate_hypothesis_seeds,
            trace=trace,
        )
        if best_for_k is None:
            _append_fit_trace_event(trace, "segment_count_invalid", k=n_segments)
            if show_progress:
                elapsed = time.perf_counter() - iter_start
                print(
                    f"[shared-fit] no valid configuration for {n_segments} segment(s) "
                    f"({elapsed:.2f}s)"
                )
            continue
        elapsed = time.perf_counter() - iter_start
        interior = np.asarray(best_for_k["interior_breaks"], dtype=float)
        rounded = np.round(interior, 3).tolist()
        trace_breaks = np.round(interior, 6).tolist()
        _append_fit_trace_event(
            trace,
            "segment_count_best",
            k=n_segments,
            score=float(best_for_k["score"]),
            interior_breaks=trace_breaks,
            elapsed_seconds=float(elapsed),
        )
        if show_progress:
            print(
                f"[shared-fit] best for {n_segments} segment(s): "
                f"{criterion.upper()}={best_for_k['score']:.3f}, "
                f"breaks={rounded} ({elapsed:.2f}s)"
            )
            # Diagnostic: plot data and per-track refits for this k
            try:
                fig, ax = plt.subplots(figsize=(10, 6))
                # y-limits across all tracks
                all_y = np.concatenate([arr[1] for arr in tracks_data.values()])
                ymin, ymax = float(np.min(all_y)), float(np.max(all_y))
                cmap = plt.get_cmap("tab10")
                for i, (track_id, arr) in enumerate(tracks_data.items()):
                    t_vals, y_vals = arr
                    color = cmap(i % 10)
                    ax.plot(t_vals, y_vals, "-", color=color, alpha=0.4)
                    # overlay refit if available
                    track_fit = best_for_k.get("track_fits", {}).get(track_id)
                    if track_fit is not None:
                        tb = np.asarray(track_fit["track_breaks"], dtype=float)
                        slopes = np.asarray(track_fit["slopes"], dtype=float)
                        by = np.asarray(track_fit.get("break_y"), dtype=float)
                        try:
                            spline_xy = sample_piecewise_linear_curve(tb, slopes, break_y=by)
                            ax.plot(spline_xy[0], spline_xy[1], color=color, linewidth=2)
                        except Exception:
                            pass
                # mark interior global break locations
                for b in interior:
                    ax.axvline(float(b), color="k", linestyle="--", alpha=0.6)
                ax.set_ylim(ymin, ymax)
                ax.set_title(f"Shared-fit diagnostic k={n_segments}, breaks={rounded}")
                ax.set_xlabel("time")
                ax.set_ylabel("y")
                plt.tight_layout()
                try:
                    # Save debug data for this k: raw tracks and fitted splines
                    dbg = {}
                    for i, (track_id, arr) in enumerate(tracks_data.items()):
                        t_vals, y_vals = arr
                        dbg[f"t_{track_id}"] = t_vals
                        dbg[f"y_{track_id}"] = y_vals
                        track_fit = best_for_k.get("track_fits", {}).get(track_id)
                        if track_fit is not None:
                            try:
                                tb = np.asarray(track_fit["track_breaks"], dtype=float)
                                slopes = np.asarray(track_fit["slopes"], dtype=float)
                                by = np.asarray(track_fit.get("break_y"), dtype=float)
                                spline_xy = sample_piecewise_linear_curve(tb, slopes, break_y=by)
                                dbg[f"spline_x_{track_id}"] = spline_xy[0]
                                dbg[f"spline_y_{track_id}"] = spline_xy[1]
                            except Exception:
                                pass
                    dbg["interior"] = interior
                    dbg["k"] = n_segments
                    dbg["score"] = float(best_for_k.get("score", np.nan))
                    # Debug-saving removed per user request; no file will be written here.
                except Exception:
                    pass
                try:
                    _display_nonblocking(fig, track_id=f"shared_k_{n_segments}")
                except Exception:
                    _PLOT_FIGURES.append(fig)
            except Exception:
                pass
        if best_overall is None or best_for_k["score"] < best_overall["score"]:
            best_overall = best_for_k

    if best_overall is None:
        raise RuntimeError("Failed to fit a shared-breakpoint piecewise-linear model.")

    best_overall["global_break_support"] = _shared_break_support(
        tracks_data,
        np.asarray(best_overall["global_breaks"], dtype=float),
    )
    if show_progress:
        elapsed = time.perf_counter() - start_time
        chosen = np.round(np.asarray(best_overall["global_breaks"], dtype=float), 3).tolist()
        print(
            f"[shared-fit] selected model: {len(chosen) - 1} segment(s), "
            f"{criterion.upper()}={best_overall['score']:.3f}, "
            f"global breaks={chosen} ({elapsed:.2f}s total)"
        )
    _append_fit_trace_event(
        trace,
        "shared_fit_selected_model",
        k=int(len(np.asarray(best_overall["global_breaks"], dtype=float)) - 1),
        global_breaks=np.round(np.asarray(best_overall["global_breaks"], dtype=float), 6).tolist(),
        score=float(best_overall["score"]),
        elapsed_seconds=float(time.perf_counter() - start_time),
    )
    return best_overall


def sample_piecewise_linear_curve(
    breaks: np.ndarray,
    slopes: np.ndarray,
    break_y: Optional[np.ndarray] = None,
    anchor: Optional[Tuple[float, float]] = None,
    points_per_segment: int = 50,
) -> np.ndarray:
    """
    Sample a continuous piecewise-linear curve from its breakpoints.

    If ``break_y`` is supplied, adjacent segments share those fitted endpoint
    values exactly. Otherwise, a single anchor point is used to propagate the
    segment slopes continuously across the full breakpoint grid.
    """
    b_arr = np.asarray(breaks, dtype=float)
    s_arr = np.asarray(slopes, dtype=float)
    if b_arr.ndim != 1 or s_arr.ndim != 1:
        raise ValueError("breaks and slopes must be 1D arrays")
    if b_arr.size != s_arr.size + 1:
        raise ValueError("breaks must have exactly one more entry than slopes")
    if points_per_segment < 2:
        raise ValueError("points_per_segment must be >= 2")

    if break_y is not None:
        y_breaks = np.asarray(break_y, dtype=float)
        if y_breaks.shape != b_arr.shape:
            raise ValueError("break_y must match the shape of breaks")
    else:
        if anchor is None:
            raise ValueError("anchor is required when break_y is not provided")
        anchor_x, anchor_y = anchor
        y_breaks = np.empty_like(b_arr)
        y_breaks[0] = float(anchor_y) + float(s_arr[0]) * (b_arr[0] - float(anchor_x))
        for i, slope in enumerate(s_arr):
            y_breaks[i + 1] = y_breaks[i] + float(slope) * (b_arr[i + 1] - b_arr[i])

    x_parts = []
    y_parts = []
    for i in range(s_arr.size):
        x0, x1 = b_arr[i], b_arr[i + 1]
        y0, y1 = y_breaks[i], y_breaks[i + 1]
        xs = np.linspace(x0, x1, points_per_segment)
        if np.isclose(x0, x1):
            ys = np.full_like(xs, y0)
        else:
            ys = np.linspace(y0, y1, points_per_segment)
        x_parts.append(xs)
        y_parts.append(ys)

    return np.vstack((np.concatenate(x_parts), np.concatenate(y_parts)))


def plot_linear_spline_fit(
    time_y: np.ndarray,
    breaks: np.ndarray,
    slopes: np.ndarray,
    break_y: Optional[np.ndarray] = None,
    ax: Optional[plt.Axes] = None,
    label_data: str = "data",
    label_fit: str = "spline fit",
    color_data: str = "C0",
    color_fit: str = "C3",
):
    """
    Plot raw data and its piecewise-linear spline fit.

    Args:
        time_y: (2, N) array of time and y.
        breaks: Breakpoints array from fit_linear_spline_auto.
        slopes: Slopes per segment from fit_linear_spline_auto.
        ax: Optional matplotlib Axes to draw on.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))
    t, y = time_y
    ax.plot(t, y, ".", ms=4, color=color_data, label=label_data)

    if break_y is None:
        _, _, break_y, _ = fit_linear_spline_with_breaks(time_y, breaks)
    spline_xy = sample_piecewise_linear_curve(breaks, slopes, break_y=break_y)
    ax.plot(spline_xy[0], spline_xy[1], color=color_fit, label=label_fit)

    ax.set_xlabel("time")
    ax.set_ylabel("y")
    ax.legend()
    ax.set_title("Piecewise-linear spline fit")
    return ax


def _tracks_data_fingerprint(tracks_data: Dict[str, np.ndarray]) -> str:
    """
    Stable fingerprint of the current track payload for exploratory-fit cache
    validation.
    """
    digest = hashlib.sha256()
    for track_id in sorted(tracks_data):
        arr = np.ascontiguousarray(np.asarray(tracks_data[track_id], dtype=float))
        digest.update(track_id.encode("utf-8"))
        digest.update(np.asarray(arr.shape, dtype=np.int64).tobytes())
        digest.update(arr.tobytes())
    return digest.hexdigest()


def _exploratory_cache_signature(
    tracks_data: Dict[str, np.ndarray],
    max_segments: int,
    criterion: str,
    min_points: int,
) -> Dict[str, object]:
    """
    Metadata describing the first-pass exploratory configuration.
    """
    return {
        "version": _EXPLORATORY_CACHE_VERSION,
        "tracks_fingerprint": _tracks_data_fingerprint(tracks_data),
        "explore_max_segments": int(max_segments + 2),
        "fallback_max_segments": int(max_segments),
        "criterion": str(criterion),
        "min_points": int(min_points),
    }


def _load_exploratory_fit_cache(
    cache_path: Optional[Path],
    tracks_data: Dict[str, np.ndarray],
    max_segments: int,
    criterion: str,
    min_points: int,
    show_progress: bool,
) -> Optional[Tuple[Dict[str, Dict[str, object]], List[np.ndarray]]]:
    """
    Load cached first-pass fits when the saved payload matches the current
    tracks and exploratory-fit settings.
    """
    if cache_path is None:
        return None

    path = Path(cache_path).expanduser().resolve()
    if not path.is_file():
        return None

    try:
        with path.open("rb") as fh:
            payload = pickle.load(fh)
    except Exception as exc:
        if show_progress:
            print(f"[exploratory-cache] failed to load {path}: {exc}")
        return None
    if not isinstance(payload, dict):
        return None

    expected_signature = _exploratory_cache_signature(
        tracks_data,
        max_segments=max_segments,
        criterion=criterion,
        min_points=min_points,
    )
    if payload.get("signature") != expected_signature:
        if show_progress:
            print(f"[exploratory-cache] cache miss for {path}: signature mismatch")
        return None

    cached_results = payload.get("results")
    if not isinstance(cached_results, dict):
        return None
    if set(cached_results.keys()) != set(tracks_data.keys()):
        return None

    try:
        fitted_breaks = [
            np.asarray(cached_results[track_id]["breakpoints"], dtype=float)
            for track_id in tracks_data
        ]
    except Exception:
        return None

    if show_progress:
        print(f"[exploratory-cache] loaded first-pass fits from {path}")
    return cached_results, fitted_breaks


def _save_exploratory_fit_cache(
    cache_path: Optional[Path],
    tracks_data: Dict[str, np.ndarray],
    max_segments: int,
    criterion: str,
    min_points: int,
    results: Dict[str, Dict[str, object]],
    show_progress: bool,
) -> None:
    """
    Persist first-pass fits so later runs can skip the exploratory stage.
    """
    if cache_path is None:
        return

    path = Path(cache_path).expanduser().resolve()
    payload = {
        "signature": _exploratory_cache_signature(
            tracks_data,
            max_segments=max_segments,
            criterion=criterion,
            min_points=min_points,
        ),
        "results": results,
    }

    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.parent / f"{path.name}.tmp"
        with tmp_path.open("wb") as fh:
            pickle.dump(payload, fh, protocol=pickle.HIGHEST_PROTOCOL)
        tmp_path.replace(path)
        if show_progress:
            print(f"[exploratory-cache] saved first-pass fits to {path}")
    except Exception as exc:
        if show_progress:
            print(f"[exploratory-cache] failed to save {path}: {exc}")


def _sample_fit_for_track(
    time_y: np.ndarray,
    breaks: np.ndarray,
    slopes: np.ndarray,
    break_y: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Sample a track-specific piecewise-linear fit, anchoring at the first
    observation when explicit breakpoint y-values are not provided.
    """
    t, y = time_y
    return sample_piecewise_linear_curve(
        breaks,
        slopes,
        break_y=break_y,
        anchor=(t[0], y[0]),
    )


def _fit_exploratory_track(
    track_id: str,
    arr: np.ndarray,
    explore_max_segments: int,
    fallback_max_segments: int,
    criterion: str,
    min_points: int,
    show_progress: bool,
) -> Tuple[Dict[str, object], np.ndarray]:
    """
    Run the first-pass exploratory fit for one track and return its public
    result payload plus the breakpoint set to feed into the shared-fit stage.
    """
    t, y = arr
    if t.size < min_points:
        endpoints = np.array([float(t[0]), float(t[-1])])
        return {
            "spline_xy": np.vstack((t, y)),
            "slopes": [],
            "breakpoints": [float(t[0]), float(t[-1])],
            "slope_se": None,
            "breakpoint_se": None,
            "break_y": [float(y[0]), float(y[-1])],
            "break_y_se": None,
        }, endpoints

    print('Running preliminary exploratory pwlf fit using AIC for track: ', track_id)
    try:
        bk_aic, slopes_aic, bk_se_aic = fit_linear_spline_auto(
            arr,
            max_segments=explore_max_segments,
            criterion="aic",
            min_points=min_points,
        )
        print(
            f"Completed exploratory AIC pwlf for track {track_id} "
            f"with {len(bk_aic)} breaks: breaks={bk_aic}, se={bk_se_aic}"
        )
    except Exception:
        print(f"Exploratory AIC pwlf failed for track {track_id}, falling back to BIC method.")
        bk_aic, slopes_aic, bk_se_aic = fit_linear_spline_auto(
            arr,
            max_segments=fallback_max_segments,
            criterion=criterion,
            min_points=min_points,
        )
        print(
            f"Completed fallback exploratory pwlf for track {track_id} "
            f"with {len(bk_aic)} breaks: breaks={bk_aic}, se={bk_se_aic}"
        )

    bk_combined = np.asarray(bk_aic, dtype=float)
    slopes = np.asarray(slopes_aic, dtype=float)
    try:
        break_y = np.empty_like(bk_combined)
        break_y[0] = float(y[0])
        for i, slope in enumerate(slopes):
            break_y[i + 1] = break_y[i] + float(slope) * (bk_combined[i + 1] - bk_combined[i])
        spline_xy = _sample_fit_for_track(arr, bk_combined, slopes, break_y=break_y)
        track_result: Dict[str, object] = {
            "spline_xy": spline_xy,
            "slopes": list(slopes),
            "breakpoints": list(bk_combined),
            "slope_se": None,
            "breakpoint_se": list(bk_se_aic) if bk_se_aic is not None else None,
            "break_y": list(break_y),
            "break_y_se": None,
        }
    except Exception:
        bk_combined = np.array([float(t[0]), float(t[-1])])
        track_result = {
            "spline_xy": np.vstack((t, y)),
            "slopes": [],
            "breakpoints": [float(t[0]), float(t[-1])],
            "slope_se": None,
            "breakpoint_se": None,
            "break_y": [float(y[0]), float(y[-1])],
            "break_y_se": None,
        }

    if show_progress:
        try:
            fig, ax = plt.subplots(figsize=(6, 3))
            spline_xy = np.asarray(track_result["spline_xy"], dtype=float)
            ax.plot(t, y, "-", color="C0", label="data")
            ax.plot(spline_xy[0], spline_xy[1], "-", color="C3", label="exploratory spline")
            ax.set_title(f"{track_id} exploratory fit: k={len(np.asarray(bk_combined, dtype=float)) - 1}")
            ax.set_xlabel("time")
            ax.set_ylabel("y")
            ax.legend()
            plt.tight_layout()
            try:
                data = {
                    "t": t,
                    "y": y,
                    "spline_x": spline_xy[0],
                    "spline_y": spline_xy[1],
                    "breaks": np.asarray(bk_combined, dtype=float),
                }
                # Debug-saving removed per user request; no file will be written here.
            except Exception:
                pass
            try:
                _ = _display_nonblocking(fig, track_id=track_id)
            except Exception:
                _PLOT_FIGURES.append(fig)
        except Exception:
            pass

    return track_result, np.asarray(bk_combined, dtype=float)


def _run_exploratory_track_fits(
    tracks_data: Dict[str, np.ndarray],
    max_segments: int,
    criterion: str,
    min_points: int,
    show_progress: bool,
) -> Tuple[Dict[str, Dict[str, object]], List[np.ndarray]]:
    """
    Run the first-pass per-track exploratory fits used both for direct output
    and for seeding the shared breakpoint optimization.
    """
    results: Dict[str, Dict[str, object]] = {}
    fitted_breaks: List[np.ndarray] = []
    explore_max_segments = max_segments + 2

    for track_id, arr in tracks_data.items():
        print('starting exploratory per track candidate generation: ', track_id)
        track_result, track_breaks = _fit_exploratory_track(
            track_id,
            arr,
            explore_max_segments=explore_max_segments,
            fallback_max_segments=max_segments,
            criterion=criterion,
            min_points=min_points,
            show_progress=show_progress,
        )
        results[track_id] = track_result
        fitted_breaks.append(np.asarray(track_breaks, dtype=float))

    return results, fitted_breaks


def _get_or_run_exploratory_track_fits(
    tracks_data: Dict[str, np.ndarray],
    max_segments: int,
    criterion: str,
    min_points: int,
    show_progress: bool,
    exploratory_cache_path: Optional[Path] = None,
) -> Tuple[Dict[str, Dict[str, object]], List[np.ndarray]]:
    """
    Reuse cached first-pass fits when available; otherwise run and optionally
    persist the exploratory stage.
    """
    cached = _load_exploratory_fit_cache(
        exploratory_cache_path,
        tracks_data,
        max_segments=max_segments,
        criterion=criterion,
        min_points=min_points,
        show_progress=show_progress,
    )
    if cached is not None:
        return cached

    results, fitted_breaks = _run_exploratory_track_fits(
        tracks_data,
        max_segments=max_segments,
        criterion=criterion,
        min_points=min_points,
        show_progress=show_progress,
    )
    _save_exploratory_fit_cache(
        exploratory_cache_path,
        tracks_data,
        max_segments=max_segments,
        criterion=criterion,
        min_points=min_points,
        results=results,
        show_progress=show_progress,
    )
    return results, fitted_breaks


def _run_shared_breakpoint_group_fit(
    tracks_data: Dict[str, np.ndarray],
    fitted_breaks: List[np.ndarray],
    max_segments: int,
    criterion: str,
    min_points: int,
    show_progress: bool,
    breakpoint_cluster_bandwidth: Optional[float] = None,
    support_window_factor: Optional[float] = 2.0,
    debug_hypothesis_seeds: Optional[Tuple[np.ndarray, ...]] = None,
    debug_validate_hypothesis_seeds: bool = True,
    trace: Optional[List[Dict[str, object]]] = None,
) -> Dict[str, Dict[str, object]]:
    """
    Run the shared-breakpoint optimization seeded by the first-pass fits and
    format the final per-track result payloads.
    """
    print('Beginning joint shared-breakpoint fitting across all tracks...')

    shared_fit = fit_shared_breakpoints_joint(
        tracks_data,
        fitted_breaks=fitted_breaks,
        max_segments=max_segments,
        criterion=criterion,
        min_points=min_points,
        show_progress=show_progress,
        breakpoint_cluster_bandwidth=breakpoint_cluster_bandwidth,
        support_window_factor=support_window_factor,
        debug_hypothesis_seeds=debug_hypothesis_seeds,
        debug_validate_hypothesis_seeds=debug_validate_hypothesis_seeds,
        trace=trace,
    )
    global_breaks = np.asarray(shared_fit["global_breaks"], dtype=float)
    global_support = np.asarray(shared_fit["global_break_support"], dtype=int)

    results: Dict[str, Dict[str, object]] = {}
    for track_id, arr in tracks_data.items():
        track_fit = shared_fit["track_fits"][track_id]
        track_breaks = np.asarray(track_fit["track_breaks"], dtype=float)
        slopes = np.asarray(track_fit["slopes"], dtype=float)
        break_y = np.asarray(track_fit["break_y"], dtype=float)
        break_y_se = track_fit["break_y_se"]
        slope_se = track_fit["slope_se"]
        spline_xy = _sample_fit_for_track(arr, track_breaks, slopes, break_y=break_y)
        results[track_id] = {
            "spline_xy": spline_xy,
            "slopes": list(slopes),
            "breakpoints": list(track_breaks),
            "slope_se": list(slope_se) if slope_se is not None else None,
            "breakpoint_se": None,
            "break_y": list(break_y),
            "break_y_se": list(break_y_se) if break_y_se is not None else None,
            "avg_breakpoints_used": list(track_breaks),
            "shared_breakpoints_used": list(track_breaks),
            "global_breakpoints": list(global_breaks),
            "global_breakpoint_support": list(global_support),
            "joint_score": float(shared_fit["score"]),
            "joint_rss": float(shared_fit["rss"]),
            "joint_n_params": int(shared_fit["n_params"]),
            "joint_criterion": criterion,
        }

    return results


def spline_fit_tracks_dict(
    tracks_input: Union[Dict[str, np.ndarray], str, Path],
    max_segments: int = 6,
    criterion: str = "bic",
    min_points: int = 4,
    use_avg_breaks: bool = False,
    cluster_on_mismatch: bool = False,
    min_segment_points: int = 2,
    show_progress: bool = False,
    exploratory_cache_path: Optional[Path] = None,
    breakpoint_cluster_bandwidth: Optional[float] = None,
    support_window_factor: Optional[float] = 2.0,
    debug_hypothesis_seeds: Optional[Tuple[np.ndarray, ...]] = None,
    debug_validate_hypothesis_seeds: bool = True,
    save_path: Optional[Union[str, Path]] = None,
) -> Dict[str, Dict[str, object]]:
    """
    Fit piecewise-linear models to every track in one dataset.

    Args:
        tracks_input: Either a dict of ``track_id -> (2, N)`` arrays
            ``[time, y]`` or a path to one extracted CSV file.
        max_segments: Max segments for pwlf search.
        criterion: 'bic' or 'aic'.
        min_points: Minimum points required to fit.
        exploratory_cache_path: Optional path for caching the expensive
            first-pass exploratory fits. When present, matching cached fits
            are reused and the exploratory stage is skipped.
        use_avg_breaks: Legacy flag. If False, fit each track independently.
            If True, estimate one shared global breakpoint set jointly across
            all tracks, then refit each track on the subset that falls inside
            its own observed time span.
        cluster_on_mismatch: Retained for backward compatibility. It is used
            only by the legacy independent-fit averaging workflow, which has
            been replaced by the joint shared-breakpoint model when
            ``use_avg_breaks=True``.
        min_segment_points: Retained for backward compatibility and metadata.
            The shared-breakpoint path no longer uses it as a hard rejection
            threshold.
        show_progress: If True, print coarse-grained progress updates during
            the joint shared-breakpoint search.
        support_window_factor: Global breakpoint-support window in units of the
            pooled median observation spacing near each shared breakpoint.
            Shared breakpoints are rejected only when they fall inside an overly
            wide pooled observation gap. Set to ``None`` to disable this global
            observability check and rely only on actual fixed-break fit success.
        debug_hypothesis_seeds: Optional manual shared-breakpoint hypotheses
            to inject when ``use_avg_breaks=True``. Each item is one interior-
            breakpoint seed, and its length maps to ``k=len(seed)+1``.
        debug_validate_hypothesis_seeds: If True, manual hypothesis seeds are
            filtered by the usual initial validity check. If False, invalid
            manual seeds are still allowed to enter refinement and are dropped
            only if no valid neighboring configuration is found.
        save_path: Optional directory where fitted nodes, fit-quality summaries,
            and metadata should be written. The directory is created if needed.

    Returns:
        Dict keyed by track_id with:
          - 'spline_xy': np.ndarray (2, M) sampled fit
          - 'slopes': list of slopes per segment
          - 'breakpoints': list of break positions (including start/end)
        If use_avg_breaks=True, results reflect refits using jointly-estimated
        shared breakpoints across all tracks.
    """
    tracks_data, source_metadata = _resolve_tracks_input(tracks_input)
    if not tracks_data:
        raise ValueError("tracks_data is empty")
    criterion = criterion.lower()
    if criterion not in {"bic", "aic"}:
        raise ValueError("criterion must be 'bic' or 'aic'")
    fit_trace: Optional[List[Dict[str, object]]] = [] if (save_path is not None and use_avg_breaks) else None
    exploratory_results, fitted_breaks = _get_or_run_exploratory_track_fits(
        tracks_data,
        max_segments=max_segments,
        criterion=criterion,
        min_points=min_points,
        show_progress=show_progress,
        exploratory_cache_path=exploratory_cache_path,
    )

    if not use_avg_breaks:
        try:
            mapping = {
                track_id: exploratory_results[track_id]["breakpoints"]
                for track_id in exploratory_results
            }
            # Debug-saving removed per user request; mapping available
            # in-memory as `mapping`.
        except Exception:
            pass
        results = exploratory_results
    else:
        results = _run_shared_breakpoint_group_fit(
            tracks_data,
            fitted_breaks=fitted_breaks,
            max_segments=max_segments,
            criterion=criterion,
            min_points=min_points,
            show_progress=show_progress,
            breakpoint_cluster_bandwidth=breakpoint_cluster_bandwidth,
            support_window_factor=support_window_factor,
            debug_hypothesis_seeds=debug_hypothesis_seeds,
            debug_validate_hypothesis_seeds=debug_validate_hypothesis_seeds,
            trace=fit_trace,
        )

    if save_path is not None:
        call_metadata = {
            "max_segments": int(max_segments),
            "criterion": criterion,
            "min_points": int(min_points),
            "use_avg_breaks": bool(use_avg_breaks),
            "cluster_on_mismatch": bool(cluster_on_mismatch),
            "min_segment_points": int(min_segment_points),
            "show_progress": bool(show_progress),
            "exploratory_cache_path": None
            if exploratory_cache_path is None
            else str(Path(exploratory_cache_path).expanduser().resolve()),
            "breakpoint_cluster_bandwidth": breakpoint_cluster_bandwidth,
            "support_window_factor": support_window_factor,
            "debug_hypothesis_seeds": None
            if debug_hypothesis_seeds is None
            else [np.asarray(seed, dtype=float).tolist() for seed in debug_hypothesis_seeds],
            "debug_validate_hypothesis_seeds": bool(debug_validate_hypothesis_seeds),
        }
        _save_spline_fit_artifacts(
            save_path=save_path,
            tracks_data=tracks_data,
            fits=results,
            call_metadata=call_metadata,
            source_metadata=source_metadata,
            trace=fit_trace,
        )

    return results


def save_avg_spline_for_file(
    output_folder: Path,
    file_stem: str,
    folder_path: Path = REFINED_DATA_DIR,
    max_segments: int = 6,
    criterion: str = "bic",
    min_points: int = 4,
    use_refined: bool = True,
    cluster_on_mismatch: bool = True,
    min_segment_points: int = 2,
    show_progress: bool = False,
    exploratory_cache_path: Optional[Path] = None,
    support_window_factor: Optional[float] = 2.0,
) -> Path:
    """
    Fit shared-breakpoint splines for a single file and save one artifact
    directory containing nodes, fit-quality metrics, and metadata.

    If ``exploratory_cache_path`` is provided, the expensive first-pass fits
    are cached there and reused on later calls when the source data and
    exploratory settings still match.

    Returns the path to the written artifact directory.
    """
    source_folder = REFINED_DATA_DIR if use_refined else Path(folder_path).expanduser().resolve()
    source_path = Path(source_folder).expanduser().resolve() / f"{file_stem}.csv"
    save_dir = Path(output_folder).expanduser().resolve() / file_stem
    spline_fit_tracks_dict(
        source_path,
        max_segments=max_segments,
        criterion=criterion,
        min_points=min_points,
        exploratory_cache_path=exploratory_cache_path,
        use_avg_breaks=True,
        cluster_on_mismatch=cluster_on_mismatch,
        min_segment_points=min_segment_points,
        show_progress=show_progress,
        support_window_factor=support_window_factor,
        save_path=save_dir,
    )
    return save_dir


if __name__ == "__main__":
    
    default_stem= "trial3"
    breakpoint_hypothesis_seeds = np.array([4, 15, 22,25,45,48,62,83])
    spline_fit_save_dir = "phys330/lab2/data_folder/fitted_unisplines"
    spline_fit_save_path = Path(spline_fit_save_dir) / default_stem
    print (spline_fit_save_path)
    default_nodes = 10
    default_min_nodes = 3
    default_min_segment_points = 2
    # Example: prune 3 highest-accel points overall, then plot the filtered result.
    raw_tracks = tracks_from_file(file_stem=default_stem,use_refined=True)
    pruned = raw_tracks
    print("Now performing joint shared-breakpoint fit across all tracks...")
    # Joint shared-breakpoint refit and plot
    fits_avg = spline_fit_tracks_dict(
        pruned,
        max_segments=default_nodes,
        criterion="bic",
        min_points=default_min_nodes,
        use_avg_breaks=True,
        min_segment_points=default_min_segment_points,
        show_progress=True,
        exploratory_cache_path=Path("phys330/lab2/src/cache/exploratory_cache.pkl"),
        breakpoint_cluster_bandwidth=1,
        debug_hypothesis_seeds=breakpoint_hypothesis_seeds,
        debug_validate_hypothesis_seeds=False,
        save_path=spline_fit_save_path,
    )
    plt.figure(figsize=(8, 5))
    for track_id, pair in pruned.items():
        t, y = pair
        plt.plot(t, y, ".", ms=3, label=f"{track_id} data")
        spline_xy = fits_avg[track_id]["spline_xy"]
        plt.plot(spline_xy[0], spline_xy[1], "-", label=f"{track_id} shared-fit")
    plt.title("Spline fits with shared global breakpoints")
    plt.xlabel("time_sec")
    plt.ylabel("y_px")
    plt.legend()
    plt.tight_layout()
    plt.show()

    """
    # Save shared-breakpoint spline fits for a refined file to disk
    try:
        output_dir = Path("phys330/lab2/data_folder/fitted_unisplines")
        out_path = save_avg_spline_for_file(
            output_folder=output_dir,
            file_stem=default_stem,
            folder_path=REFINED_DATA_DIR,
            max_segments=default_nodes,
            criterion="bic",
            min_points=default_min_nodes,
            use_refined=True,
            min_segment_points=default_min_segment_points,
            show_progress=True,
        )
        print(f"Saved shared-breakpoint spline fit to {out_path}")
    except ImportError:
        print("pwlf not installed; skipping spline file export.")
    """
