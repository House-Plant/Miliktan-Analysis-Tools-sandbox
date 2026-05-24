"""
Small debugging helpers for inspecting cached exploratory breakpoint data.
"""

from pathlib import Path
import sys
from typing import Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import MeanShift


CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from data_analyzer import _candidate_break_grid, _load_exploratory_fit_cache
from data_extractor import REFINED_DATA_DIR, tracks_from_file


def _top_meanshift_centers(
    breakpoints: np.ndarray,
    bandwidth: float,
    max_centers: int,
) -> np.ndarray:
    """
    Return the top MeanShift cluster centers ranked by cluster population.
    """
    bp = np.asarray(breakpoints, dtype=float)
    if bp.size == 0:
        return np.array([], dtype=float)

    ms = MeanShift(bandwidth=float(bandwidth), bin_seeding=True)
    ms.fit(bp.reshape(-1, 1))
    cluster_centers = ms.cluster_centers_.flatten()
    labels = ms.labels_
    unique_labels, counts = np.unique(labels, return_counts=True)
    sorted_indices = np.argsort(counts)[::-1][:max_centers]
    return np.sort(cluster_centers[unique_labels[sorted_indices]])


def _candidate_breakpoint_density(
    breakpoints: np.ndarray,
    x_min: float,
    x_max: float,
    num_points: int = 400,
    bandwidth: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute a simple Gaussian-kernel density estimate for breakpoint times
    using only NumPy.
    """
    bp = np.asarray(breakpoints, dtype=float)
    if bp.size == 0:
        return np.linspace(x_min, x_max, num_points), np.zeros(num_points, dtype=float)

    xs = np.linspace(x_min, x_max, num_points)
    bandwidth = _resolve_density_bandwidth(bp, x_min=x_min, x_max=x_max, bandwidth=bandwidth)
    diffs = (xs[:, None] - bp[None, :]) / float(bandwidth)
    kernel = np.exp(-0.5 * diffs * diffs) / (np.sqrt(2.0 * np.pi) * float(bandwidth))
    density = np.mean(kernel, axis=1)
    return xs, density


def _resolve_density_bandwidth(
    breakpoints: np.ndarray,
    x_min: float,
    x_max: float,
    bandwidth: Optional[float] = None,
) -> float:
    """
    Resolve the scalar bandwidth used by the breakpoint density overlay.
    """
    bp = np.asarray(breakpoints, dtype=float)
    span = max(float(x_max - x_min), 1e-12)
    if bandwidth is None:
        if bp.size > 1:
            sigma = float(np.std(bp, ddof=1))
            bandwidth = 1.06 * sigma * (bp.size ** (-1.0 / 5.0))
        else:
            bandwidth = 0.0
        bandwidth = max(float(bandwidth), 0.02 * span, 1e-3)
    return float(bandwidth)


def load_cached_candidate_breakpoints(
    file_stem: str,
    cache_path: Path = CURRENT_DIR / "cache" / "exploratory_cache.pkl",
    folder_path: Path = REFINED_DATA_DIR,
    max_segments: int = 10,
    criterion: str = "bic",
    min_points: int = 3,
    use_refined: bool = True,
) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
    """
    Load track data plus the candidate shared-breakpoint grid reconstructed
    from a previously saved exploratory-fit cache.
    """
    tracks_data = tracks_from_file(
        file_stem=file_stem,
        data_folder=folder_path,
        use_refined=use_refined,
    )
    cached = _load_exploratory_fit_cache(
        cache_path=cache_path,
        tracks_data=tracks_data,
        max_segments=max_segments,
        criterion=criterion,
        min_points=min_points,
        show_progress=False,
    )
    if cached is None:
        raise RuntimeError(
            f"No valid exploratory cache could be loaded from {Path(cache_path).expanduser().resolve()}"
        )

    _, fitted_breaks = cached
    candidate_breaks = _candidate_break_grid(tracks_data, fitted_breaks)
    return tracks_data, candidate_breaks


def print_top_meanshift_breakpoints(
    file_stem: str,
    track_id: str = "track3",
    cache_path: Path = CURRENT_DIR / "cache" / "exploratory_cache.pkl",
    folder_path: Path = REFINED_DATA_DIR,
    max_segments: int = 10,
    criterion: str = "bic",
    min_points: int = 3,
    use_refined: bool = True,
    clip_to_track_span: bool = True,
    density_bandwidth: Optional[float] = None,
    n_breakpoints: int = 9,
) -> np.ndarray:
    """
    Print the top MeanShift breakpoint centers for the selected track view.
    """
    tracks_data, candidate_breaks = load_cached_candidate_breakpoints(
        file_stem=file_stem,
        cache_path=cache_path,
        folder_path=folder_path,
        max_segments=max_segments,
        criterion=criterion,
        min_points=min_points,
        use_refined=use_refined,
    )
    if track_id not in tracks_data:
        available = ", ".join(sorted(tracks_data))
        raise KeyError(f"Unknown track_id {track_id!r}. Available tracks: {available}")

    t = np.asarray(tracks_data[track_id][0], dtype=float)
    visible_breaks = np.asarray(candidate_breaks, dtype=float)
    if clip_to_track_span:
        visible_breaks = visible_breaks[
            (visible_breaks > float(np.min(t))) & (visible_breaks < float(np.max(t)))
        ]

    resolved_bandwidth = _resolve_density_bandwidth(
        visible_breaks,
        x_min=float(np.min(t)),
        x_max=float(np.max(t)),
        bandwidth=density_bandwidth,
    )
    top_centers = _top_meanshift_centers(
        visible_breaks,
        bandwidth=resolved_bandwidth,
        max_centers=n_breakpoints,
    )
    rounded = np.round(top_centers, 6).tolist()
    print(f"Top {n_breakpoints} MeanShift breakpoints for {file_stem}/{track_id}: {rounded}")
    return top_centers


def plot_track_with_candidate_breakpoints(
    file_stem: str,
    track_id: str = "track3",
    cache_path: Path = CURRENT_DIR / "cache" / "exploratory_cache.pkl",
    folder_path: Path = REFINED_DATA_DIR,
    max_segments: int = 10,
    criterion: str = "bic",
    min_points: int = 3,
    use_refined: bool = True,
    clip_to_track_span: bool = True,
    show_density: bool = True,
    density_bandwidth: Optional[float] = None,
    ax: Optional[plt.Axes] = None,
    show: bool = True,
) -> plt.Axes:
    """
    Plot one track as a line graph and overlay candidate shared breakpoints as
    dotted vertical lines.
    """
    tracks_data, candidate_breaks = load_cached_candidate_breakpoints(
        file_stem=file_stem,
        cache_path=cache_path,
        folder_path=folder_path,
        max_segments=max_segments,
        criterion=criterion,
        min_points=min_points,
        use_refined=use_refined,
    )

    if track_id not in tracks_data:
        available = ", ".join(sorted(tracks_data))
        raise KeyError(f"Unknown track_id {track_id!r}. Available tracks: {available}")

    t, y = tracks_data[track_id]
    visible_breaks = np.asarray(candidate_breaks, dtype=float)
    if clip_to_track_span:
        visible_breaks = visible_breaks[
            (visible_breaks > float(np.min(t))) & (visible_breaks < float(np.max(t)))
        ]

    if ax is None:
        _, ax = plt.subplots(figsize=(10, 4))

    ax.plot(t, y, "-", color="C0", linewidth=1.5, label=f"{track_id} data")
    for i, bp in enumerate(visible_breaks):
        ax.axvline(
            float(bp),
            color="C3",
            linestyle=":",
            linewidth=1.0,
            alpha=0.8,
            label="candidate breakpoints" if i == 0 else None,
        )

    if visible_breaks.size > 0:
        resolved_bandwidth = _resolve_density_bandwidth(
            visible_breaks,
            x_min=float(np.min(t)),
            x_max=float(np.max(t)),
            bandwidth=density_bandwidth,
        )
        top_centers = _top_meanshift_centers(
            visible_breaks,
            bandwidth=resolved_bandwidth,
            max_centers=10,
        )
        for i, center in enumerate(top_centers):
            ax.axvline(
                float(center),
                color="C2",
                linestyle=":",
                linewidth=2.4,
                alpha=0.95,
                label="top MeanShift centers" if i == 0 else None,
            )
    else:
        resolved_bandwidth = density_bandwidth

    density_ax = None
    if show_density and visible_breaks.size > 0:
        density_ax = ax.twinx()
        density_x, density_y = _candidate_breakpoint_density(
            visible_breaks,
            x_min=float(np.min(t)),
            x_max=float(np.max(t)),
            bandwidth=resolved_bandwidth,
        )
        density_ax.fill_between(
            density_x,
            density_y,
            0.0,
            color="C3",
            alpha=0.16,
            label="candidate density",
        )
        density_ax.plot(
            density_x,
            density_y,
            color="C3",
            alpha=0.55,
            linewidth=1.5,
        )
        density_ax.set_ylabel("candidate density", color="C3")
        density_ax.tick_params(axis="y", colors="C3")
        density_ax.spines["right"].set_color("C3")

    ax.set_title(f"{file_stem}: {track_id} with cached candidate breakpoints")
    ax.set_xlabel("time")
    ax.set_ylabel("y")
    handles, labels = ax.get_legend_handles_labels()
    if density_ax is not None:
        density_handles, density_labels = density_ax.get_legend_handles_labels()
        handles.extend(density_handles)
        labels.extend(density_labels)
    ax.legend(handles, labels)
    ax.grid(alpha=0.2)

    if show:
        plt.tight_layout()
        plt.show()

    return ax


if __name__ == "__main__":
    print_top_meanshift_breakpoints(
        file_stem="trial3",
        track_id="track3",
        density_bandwidth=0.3,
        n_breakpoints=9,
    )
    plot_track_with_candidate_breakpoints(file_stem="trial3", track_id="track3", density_bandwidth=0.3)
