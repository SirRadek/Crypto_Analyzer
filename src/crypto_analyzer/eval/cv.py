from __future__ import annotations

import numpy as np
import pandas as pd


def purged_walkforward_splits(
    index: pd.DatetimeIndex, n_splits: int, embargo_min: int
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Build purged walk-forward cross-validation splits.

    Parameters
    ----------
    index:
        Chronologically ordered :class:`pandas.DatetimeIndex` describing the
        observations that should be split.
    n_splits:
        Number of walk-forward folds to generate.
    embargo_min:
        Embargo window (in minutes) applied on both sides of each test fold.

    Returns
    -------
    list[tuple[np.ndarray, np.ndarray]]
        Sequence of ``(train_idx, test_idx)`` pairs referencing the positional
        indices of the provided ``index``.  Each fold respects the
        chronological order (``train`` precedes ``test``) while purging the
        embargo window around the test set from the training samples.
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

    boundaries = np.linspace(0, n_samples, n_splits + 2, dtype=int)
    splits: list[tuple[np.ndarray, np.ndarray]] = []
    embargo_windows: list[tuple[pd.Timestamp, pd.Timestamp]] = []

    for i in range(n_splits):
        train_end = boundaries[i + 1]
        test_start = train_end
        test_end = boundaries[i + 2]

        if test_end <= test_start:
            continue

        test_idx = np.arange(test_start, test_end)
        test_start_time = index[test_idx[0]]
        test_end_time = index[test_idx[-1]]

        window_start = test_start_time - embargo_delta
        window_end = test_end_time + embargo_delta

        train_mask = np.zeros(n_samples, dtype=bool)
        if train_end > 0:
            train_mask[:test_start] = True
            for start, end in embargo_windows + [(window_start, window_end)]:
                mask = (index >= start) & (index <= end)
                train_mask &= ~mask

        train_idx = np.nonzero(train_mask)[0]
        if train_idx.size == 0:
            embargo_windows.append((window_start, window_end))
            continue

        splits.append((train_idx, test_idx))
        embargo_windows.append((window_start, window_end))

    return splits
