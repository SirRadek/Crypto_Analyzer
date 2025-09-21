import json
from pathlib import Path

import pandas as pd

from crypto_analyzer.eval.cv import purged_walkforward_splits
from crypto_analyzer.utils.splitting import PurgedWalkForwardSplit


def test_purged_walkforward_splits_respect_order_and_embargo():
    index = pd.date_range("2024-01-01", periods=120, freq="h", tz="UTC")
    embargo_min = 180

    splits = purged_walkforward_splits(index, n_splits=5, embargo_min=embargo_min)

    assert splits, "Expected to receive at least one purged walk-forward split"

    embargo_delta = pd.Timedelta(minutes=embargo_min)
    previous_test_end = index[0]
    seen_windows: list[tuple[pd.Timestamp, pd.Timestamp]] = []

    for train_idx, test_idx in splits:
        assert train_idx.size > 0
        assert test_idx.size > 0

        train_times = index[train_idx]
        test_times = index[test_idx]

        assert (train_times < test_times.min()).all(), "Training indices must precede testing ones"

        window_start = test_times[0] - embargo_delta
        window_end = test_times[-1] + embargo_delta

        for embargo_start, embargo_end in seen_windows + [(window_start, window_end)]:
            overlap = (train_times >= embargo_start) & (train_times <= embargo_end)
            assert not overlap.any(), "Training data must respect embargo windows"

        assert test_times[0] >= previous_test_end, "Test windows must move forward in time"

        previous_test_end = test_times[-1]
        seen_windows.append((window_start, window_end))


def test_purged_walkforward_split_applies_purge_and_embargo():
    timestamps = pd.date_range("2024-01-01", periods=72, freq="h", tz="UTC")
    df = pd.DataFrame({"timestamp": timestamps})

    purge_minutes = 120
    embargo_minutes = 90

    splitter = PurgedWalkForwardSplit(
        train_span_days=2,
        test_span_days=1,
        step_days=1,
        min_train_days=2,
        purge_minutes=purge_minutes,
        embargo_minutes=embargo_minutes,
    )

    splits = list(splitter.split(df))
    assert splits, "Expected purged walk-forward splitter to produce folds"

    purge_delta = pd.Timedelta(minutes=purge_minutes)
    embargo_delta = pd.Timedelta(minutes=embargo_minutes)
    previous_test_end: pd.Timestamp | None = None

    for train_idx, test_idx in splits:
        train_times = timestamps[train_idx]
        test_times = timestamps[test_idx]

        assert (train_times < test_times.min()).all()
        assert train_times.max() <= test_times.min() - purge_delta

        if previous_test_end is not None:
            assert train_times.min() >= previous_test_end + embargo_delta

        previous_test_end = test_times.max()


def test_purged_walkforward_exports_metadata(tmp_path: Path):
    index = pd.date_range("2024-01-01", periods=24, freq="h", tz="UTC")
    run_id = "unit"
    embargo = 30

    splits = purged_walkforward_splits(
        index,
        n_splits=3,
        embargo_min=embargo,
        run_id=run_id,
        reports_dir=tmp_path,
    )

    cv_path = tmp_path / f"cv_{run_id}.json"
    assert cv_path.exists(), "Expected purged walk-forward splits to be exported"

    payload = json.loads(cv_path.read_text(encoding="utf-8"))
    assert len(payload) == len(splits)

    first_fold = payload[0]
    assert first_fold["fold"] == 0
    assert first_fold["embargo_min"] == embargo
    assert first_fold["train_start"] <= first_fold["train_end"]
    assert first_fold["test_start"] <= first_fold["test_end"]
