import pandas as pd

from crypto_analyzer.utils.splitting import WalkForwardSplit


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
