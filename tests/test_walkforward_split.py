import pandas as pd

from crypto_analyzer.eval.cv import purged_walkforward_splits
from crypto_analyzer.utils.splitting import PurgedWalkForwardSplit, WalkForwardSplit


def test_walkforward_split_preserves_temporal_order():
    ts = pd.date_range("2024-01-01", periods=96, freq="h", tz="UTC")
    df = pd.DataFrame({"timestamp": ts})

    splitter = WalkForwardSplit(
        train_span_days=2, test_span_days=1, step_days=1, min_train_days=2
    )

    folds = list(splitter.split(df))
    assert folds, "Expected at least one fold to be generated"

    last_test_end = -1
    for train_idx, test_idx in folds:
        assert len(train_idx) > 0 and len(test_idx) > 0
        assert train_idx.max() < test_idx.min()
        assert train_idx.max() > last_test_end
        last_test_end = test_idx.max()

    covered = sorted({idx for _, test_idx in folds for idx in test_idx})
    assert covered == sorted(covered)


def test_purged_walkforward_respects_embargo_and_purge():
    ts = pd.date_range("2024-01-01", periods=48, freq="h", tz="UTC")
    df = pd.DataFrame({"timestamp": ts})

    splitter = PurgedWalkForwardSplit(
        train_span_days=1,
        test_span_days=1,
        step_days=1,
        min_train_days=1,
        purge_minutes=120,
        embargo_minutes=60,
    )

    splits = list(splitter.split(df))
    assert splits, "Expected purged splitter to produce folds"
    for train_idx, test_idx in splits:
        assert train_idx.max() < test_idx.min()
    if len(splits) > 1:
        for (_, test_idx), (next_train_idx, _) in zip(splits, splits[1:]):
            assert test_idx.max() < next_train_idx.min()


def test_purged_walkforward_split_function_respects_embargo():
    index = pd.date_range("2024-01-01", periods=72, freq="h", tz="UTC")
    embargo_min = 180

    splits = purged_walkforward_splits(index, n_splits=4, embargo_min=embargo_min)
    assert splits, "Expected purged walk-forward splitter to return folds"

    embargo_delta = pd.Timedelta(minutes=embargo_min)
    seen_windows: list[tuple[pd.Timestamp, pd.Timestamp]] = []

    for train_idx, test_idx in splits:
        assert train_idx.size > 0 and test_idx.size > 0
        assert train_idx.max() < test_idx.min()

        train_times = index[train_idx]
        test_times = index[test_idx]

        window_start = test_times[0] - embargo_delta
        window_end = test_times[-1] + embargo_delta

        assert not ((train_times >= window_start) & (train_times <= window_end)).any()
        for start, end in seen_windows:
            assert not ((train_times >= start) & (train_times <= end)).any()

        seen_windows.append((window_start, window_end))
