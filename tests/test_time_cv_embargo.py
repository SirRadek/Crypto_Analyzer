from ml.time_cv import time_folds


def test_time_folds_embargo() -> None:
    n = 200
    folds = list(time_folds(n, n_splits=3, embargo=24))
    for train_idx, test_idx in folds:
        if len(train_idx) == 0:
            continue
        assert train_idx.max() <= test_idx.min() - 24
