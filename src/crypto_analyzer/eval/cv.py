from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


def purged_walkforward_splits(
    index: pd.DatetimeIndex,
    n_splits: int,
    embargo_min: int,
    *,
    run_id: str | None = None,
    reports_dir: str | Path = "reports",
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Build purged walk-forward cross-validation splits.

    The splitter respects temporal ordering (training observations always
    precede their corresponding test fold) and applies a symmetric embargo
    around every test window.  Observations falling inside the embargo window
    are removed from all training folds to avoid leakage.
    """

    if n_splits < 1:
        raise ValueError("n_splits must be at least 1")

    if not isinstance(index, pd.DatetimeIndex):
        index = pd.DatetimeIndex(index)

    if index.hasnans:
        raise ValueError("DatetimeIndex must not contain NaT values")

    if not index.is_monotonic_increasing:
        raise ValueError("DatetimeIndex must be sorted in ascending order")

    n_samples = len(index)
    if n_samples <= n_splits:
        raise ValueError("Not enough samples to generate the requested splits")

    embargo_minutes = max(0, int(embargo_min))
    embargo_delta = pd.Timedelta(minutes=embargo_minutes)

    # Divide the index into n_splits test segments + one initial training block.
    # ``np.array_split`` keeps the chronological order and balances the segment
    # sizes while ensuring every test segment receives at least one observation
    # when possible.
    segments = np.array_split(np.arange(n_samples, dtype=int), n_splits + 1)

    splits: list[tuple[np.ndarray, np.ndarray]] = []
    export_records: list[dict[str, object]] = []
    embargo_windows: list[tuple[pd.Timestamp, pd.Timestamp]] = []

    for fold in range(n_splits):
        test_idx = segments[fold + 1]
        if test_idx.size == 0:
            continue

        test_start_time = index[test_idx[0]]
        test_end_time = index[test_idx[-1]]

        window_start = test_start_time - embargo_delta
        window_end = test_end_time + embargo_delta

        train_mask = np.zeros(n_samples, dtype=bool)
        for segment in segments[: fold + 1]:
            if segment.size:
                train_mask[segment] = True

        # Enforce the chronological constraint explicitly and purge previous
        # and current embargo windows from the training candidates.
        train_mask &= index < test_start_time
        for start, end in embargo_windows + [(window_start, window_end)]:
            mask = (index >= start) & (index <= end)
            train_mask &= ~mask

        train_idx = np.flatnonzero(train_mask)
        if train_idx.size == 0:
            embargo_windows.append((window_start, window_end))
            continue

        splits.append((train_idx, test_idx))

        record_fold = len(splits) - 1
        export_records.append(
            {
                "fold": record_fold,
                "train_start": index[train_idx[0]].isoformat(),
                "train_end": index[train_idx[-1]].isoformat(),
                "test_start": index[test_idx[0]].isoformat(),
                "test_end": index[test_idx[-1]].isoformat(),
                "embargo_min": embargo_minutes,
            }
        )
        embargo_windows.append((window_start, window_end))

    if run_id is not None:
        reports_path = Path(reports_dir)
        reports_path.mkdir(parents=True, exist_ok=True)
        export_path = reports_path / f"cv_{run_id}.json"
        export_path.write_text(
            json.dumps(export_records, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    return splits
