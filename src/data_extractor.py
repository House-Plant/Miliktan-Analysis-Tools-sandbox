"""
Helpers for loading the extracted CSV datasets used in lab 2.

This module provides two small utilities:
  * ``load_extracted_csvs`` loads every CSV inside ``data_folder/extracted_raw``
    into pandas DataFrames.
  * ``print_column_labels`` prints the column headers for a given dataset
    (either a DataFrame or a path to a CSV file).
"""

from pathlib import Path
from typing import Dict, Union, Optional, List, Tuple

import pandas as pd
import numpy as np


# Resolve the lab2 root based on this file's location.
ROOT_DIR = Path(__file__).resolve().parents[1]
# Default folders containing the extracted CSV files.
DEFAULT_DATA_DIR = ROOT_DIR / "data_folder" / "extracted_raw"
REFINED_DATA_DIR = ROOT_DIR / "data_folder" / "extracted_refined"


def load_extracted_csvs(folder: Union[str, Path] = DEFAULT_DATA_DIR) -> Dict[str, pd.DataFrame]:
    """
    Load all CSV files from ``folder`` into pandas DataFrames.

    Args:
        folder: Directory containing the extracted CSV files. Defaults to
                ``data_folder/extracted_raw`` relative to this file.

    Returns:
        Dict mapping file stems (e.g., ``trial_1_pointtrack``) to DataFrames.
    """

    folder = Path(folder).expanduser().resolve()
    if not folder.is_dir():
        raise NotADirectoryError(f"{folder} is not a directory")

    data: Dict[str, pd.DataFrame] = {}
    for csv_path in sorted(folder.glob("*.csv")):
        if not csv_path.is_file():
            continue
        data[csv_path.stem] = pd.read_csv(csv_path)
    return data


def print_column_labels(dataset: Union[pd.DataFrame, str, Path]) -> None:
    """
    Print the column labels for a dataset.

    Args:
        dataset: Either a pandas DataFrame or a path to a CSV file.
    """

    if isinstance(dataset, (str, Path)):
        csv_path = Path(dataset).expanduser().resolve()
        df = pd.read_csv(csv_path)
        name = csv_path.name
    else:
        df = dataset
        name = getattr(dataset, "name", "DataFrame")

    cols = list(df.columns)
    print(f"{name} columns ({len(cols)}): {cols}")


def split_by_track_id(dataset: Union[pd.DataFrame, str, Path]) -> Dict[str, pd.DataFrame]:
    """
    Split a point-tracking dataset into per-track dicts with time and xy pixels.

    Args:
        dataset: DataFrame or path to a CSV file containing columns
                 ``track_id``, ``time_sec``, ``x_px``, and ``y_px``.

    Returns:
        Dict mapping each track_id (e.g., ``"track1"``) to a dict with:
            - ``"xydata_px"``: np.ndarray shaped (2, N) with x then y pixels.
            - ``"time_sec"``:  np.ndarray shaped (N,) of timestamps.
    """

    if isinstance(dataset, (str, Path)):
        df = pd.read_csv(Path(dataset).expanduser().resolve())
    else:
        df = dataset

    required = {"track_id", "time_sec", "x_px", "y_px"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"Missing required columns: {missing}")

    grouped: Dict[str, Dict[str, np.ndarray]] = {}
    for track_id, group in df.groupby("track_id"):
        # Sort by time so xy and time stay aligned chronologically.
        track_df = group.sort_values("time_sec").reset_index(drop=True)
        xydata_px = track_df[["x_px", "y_px"]].to_numpy().T  # shape (2, N)
        time_sec = track_df["time_sec"].to_numpy()
        grouped[str(track_id)] = {
            "xydata_px": xydata_px,
            "time_sec": time_sec,
        }
    return grouped


def split_refined_by_track(dataset: Union[pd.DataFrame, str, Path]) -> Dict[str, np.ndarray]:
    """
    Handle refined CSVs that contain columns: track_id, time_sec, y_px.

    Returns:
        Dict mapping track_id -> array shaped (2, N) with [time_sec, y_px].
    """
    if isinstance(dataset, (str, Path)):
        df = pd.read_csv(Path(dataset).expanduser().resolve())
    else:
        df = dataset

    required = {"track_id", "time_sec", "y_px"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"Missing required columns: {missing}")

    result: Dict[str, np.ndarray] = {}
    for track_id, group in df.groupby("track_id"):
        track_df = group.sort_values("time_sec").reset_index(drop=True)
        time_sec = track_df["time_sec"].to_numpy()
        y_px = track_df["y_px"].to_numpy()
        result[str(track_id)] = np.vstack((time_sec, y_px))
    return result


def load_all_tracks(folder: Union[str, Path] = DEFAULT_DATA_DIR) -> Dict[str, Dict[str, Dict[str, np.ndarray]]]:
    """
    Load every CSV in ``folder`` and split each into per-track dicts.

    Args:
        folder: Directory containing the extracted CSV files. Defaults to
                ``data_folder/extracted_raw`` relative to this file.

    Returns:
        Nested dict shaped like::

            {
                \"trial_1_pointtrack\": {\"track1\": {\"xydata_px\": ..., \"time_sec\": ...}, ...},
                \"trial_2_pointtrack\": {...},
                ...
            }
    """
    datasets = load_extracted_csvs(folder)
    return {name: split_by_track_id(df) for name, df in datasets.items()}


def tracks_from_file(
    file_stem: Optional[str] = None,
    data_folder: Path = DEFAULT_DATA_DIR,
    use_refined: bool = False,
) -> Dict[str, np.ndarray]:
    """
    Load one CSV (by stem) and return each track's y_px vs time.

    Returns:
        Dict mapping track_id -> array shaped (2, N) with [time_sec, y_px].
    """
    if use_refined:
        data_folder = REFINED_DATA_DIR
        refined = load_extracted_csvs(data_folder)
        if not refined:
            raise FileNotFoundError(f"No CSV files found in {data_folder}")
        if file_stem is None:
            file_stem = next(iter(refined.keys()))
        if file_stem not in refined:
            raise KeyError(f"{file_stem} not found; available: {list(refined.keys())}")
        return split_refined_by_track(refined[file_stem])
    else:
        all_tracks = load_all_tracks(data_folder)
        if not all_tracks:
            raise FileNotFoundError(f"No CSV files found in {data_folder}")

        if file_stem is None:
            file_stem = next(iter(all_tracks.keys()))
        if file_stem not in all_tracks:
            raise KeyError(f"{file_stem} not found; available: {list(all_tracks.keys())}")

        tracks = all_tracks[file_stem]

        time_y_data: Dict[str, np.ndarray] = {}
        for track_id, payload in tracks.items():
            y_px = payload["xydata_px"][1]  # second row is y
            time_sec = payload["time_sec"]
            pair = np.vstack((time_sec, y_px))  # shape (2, N)
            time_y_data[track_id] = pair
        return time_y_data


def tracks_from_folder(
    folder: Path = DEFAULT_DATA_DIR,
    file_stems: Optional[List[str]] = None,
    use_refined: bool = False,
) -> Tuple[Dict[str, Dict[str, np.ndarray]], Dict[str, np.ndarray]]:
    """
    Load multiple CSVs and return:
      1) A dict keyed by file stem -> track dicts (tracks_from_file format).
      2) A merged dict of all tracks across files with dynamic renaming.

    If ``file_stems`` is provided, only those files (in the given order) are
    processed. Otherwise, all CSVs in the folder are used in sorted order.

    Merged naming: file index starts at 1 in the chosen order. Track IDs that
    start with \"track\" get their numeric suffix kept; e.g., file 1's \"track2\"
    becomes \"track1-2\". Other track ids use their raw id after the dash.
    """
    if use_refined:
        folder = REFINED_DATA_DIR

    # Determine file list
    if file_stems is None:
        file_stems = sorted(load_extracted_csvs(folder).keys())
    if not file_stems:
        raise FileNotFoundError(f"No CSV files found in {folder}")

    by_file: Dict[str, Dict[str, np.ndarray]] = {}
    merged: Dict[str, np.ndarray] = {}

    for idx, stem in enumerate(file_stems, start=1):
        tracks = tracks_from_file(file_stem=stem, data_folder=folder, use_refined=use_refined)
        by_file[stem] = tracks
        for track_id, arr in tracks.items():
            suffix = track_id[5:] if track_id.lower().startswith("track") else track_id
            new_key = f"track{idx}-{suffix}"
            merged[new_key] = arr

    return by_file, merged


def save_tracks_to_folder(
    output_folder: Path,
    source_folder: Path = DEFAULT_DATA_DIR,
    file_stems: Optional[List[str]] = None,
) -> List[Path]:
    """
    Load tracks from source_folder and write per-file tidy CSVs to output_folder.

    Output files are named ``trial1.csv``, ``trial2.csv``, ... in the order
    determined by ``file_stems`` (if provided) or sorted source files.

    Each output CSV has columns: ``track_id``, ``time_sec``, ``y_px``.
    """
    output_folder = Path(output_folder).expanduser().resolve()
    output_folder.mkdir(parents=True, exist_ok=True)

    # Load tracks in requested order.
    if file_stems is None:
        file_stems = sorted(load_extracted_csvs(source_folder).keys())
    by_file, _ = tracks_from_folder(folder=source_folder, file_stems=file_stems)

    written_paths: List[Path] = []
    for idx, stem in enumerate(file_stems, start=1):
        tracks = by_file[stem]
        rows = []
        for track_id, arr in tracks.items():
            t, y = arr
            rows.append(
                pd.DataFrame(
                    {
                        "track_id": track_id,
                        "time_sec": t,
                        "y_px": y,
                    }
                )
            )
        df_out = pd.concat(rows, ignore_index=True)
        out_path = output_folder / f"trial{idx}.csv"
        df_out.to_csv(out_path, index=False)
        written_paths.append(out_path)
    return written_paths


if __name__ == "__main__":
    pass