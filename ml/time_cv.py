from __future__ import annotations

from collections.abc import Iterator

import numpy as np


def time_folds(
    n_samples: int, n_splits: int = 5, embargo: int = 24
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Yield time-based train/test indices with embargo."""
    indices = np.arange(n_samples)
    fold_size = n_samples // (n_splits + 1)
    for i in range(1, n_splits + 1):
        test_start = i * fold_size
        test_end = min((i + 1) * fold_size, n_samples)
        train_end = max(0, test_start - embargo)
        train_idx = indices[:train_end]
        test_idx = indices[test_start:test_end]
        if len(test_idx) == 0:
            break
        yield train_idx, test_idx
