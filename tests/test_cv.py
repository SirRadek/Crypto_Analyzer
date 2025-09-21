from __future__ import annotations

import pandas as pd

from crypto_analyzer.eval.cv import purged_walkforward_splits


def test_purged_walkforward_respects_embargo():
    index = pd.date_range("2024-01-01", periods=60, freq="5T", tz="UTC")
    splits = purged_walkforward_splits(index, n_splits=4, embargo_min=15)
    assert len(splits) >= 2
    for train_idx, test_idx in splits:
        if len(train_idx) == 0 or len(test_idx) == 0:
            continue
        assert train_idx.max() < test_idx.min()
        train_times = index[train_idx]
        test_times = index[test_idx]
        embargo_start = test_times.min() - pd.Timedelta(minutes=15)
        embargo_end = test_times.max() + pd.Timedelta(minutes=15)
        assert not ((train_times >= embargo_start) & (train_times <= embargo_end)).any()
