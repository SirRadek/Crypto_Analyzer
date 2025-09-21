from __future__ import annotations

import pandas as pd
import pytest

from crypto_analyzer.utils.merge import merge_left_labeled, validate_left_label_alignment


def test_merge_left_labeled_respects_left_label_alignment() -> None:
    target_ts = pd.date_range("2024-01-01", periods=4, freq="5min", tz="UTC")
    targets = pd.DataFrame(
        {
            "timestamp_target_open": target_ts,
            "target": [0.1, -0.2, 0.05, 0.4],
        }
    )

    feature_ts = [
        target_ts[0] - pd.Timedelta(minutes=5),
        target_ts[1],
        target_ts[3],
    ]
    features = pd.DataFrame(
        {
            "timestamp_feature": feature_ts,
            "alpha": [1.0, 2.0, 3.0],
        }
    )

    merged = merge_left_labeled(targets, features)

    expected_alignment = [
        feature_ts[0],
        feature_ts[1],
        feature_ts[1],
        feature_ts[2],
    ]
    assert list(merged["timestamp_feature"].to_list()) == expected_alignment

    valid = merged["timestamp_feature"].notna()
    aligned_features = merged.loc[valid, "timestamp_feature"]
    aligned_targets = merged.loc[valid, "timestamp_target_open"]
    assert (aligned_features <= aligned_targets).all()


def test_validate_left_label_alignment_detects_future_features() -> None:
    df = pd.DataFrame(
        {
            "timestamp_target_open": pd.to_datetime(
                ["2024-01-01T00:00:00Z", "2024-01-01T00:05:00Z"], utc=True
            ),
            "timestamp_feature": pd.to_datetime(
                ["2024-01-01T00:00:00Z", "2024-01-01T00:06:00Z"], utc=True
            ),
        }
    )

    with pytest.raises(AssertionError):
        validate_left_label_alignment(df)

