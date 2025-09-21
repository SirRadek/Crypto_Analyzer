from __future__ import annotations

import warnings

import pandas as pd
import pytest

from crypto_analyzer.utils.time import (
    assert_no_future_leak,
    ensure_utc_series,
    resample_left_label_last,
)

warnings.filterwarnings("error", category=FutureWarning, module="pandas")


def test_ensure_utc_series_normalizes_timezones() -> None:
    series = pd.Series(["2024-01-01T00:00:00", "2024-01-01T00:05:00Z"])
    coerced = ensure_utc_series(series, column_name="timestamp")
    assert str(coerced.dt.tz) == "UTC"

    aware = pd.Series(pd.date_range("2024-01-01", periods=2, freq="5min", tz="UTC"))
    coerced_aware = ensure_utc_series(aware, column_name="timestamp")
    pd.testing.assert_series_equal(coerced_aware, aware)


def test_resample_left_label_last_respects_left_label_semantics() -> None:
    ts = pd.to_datetime(
        [
            "2024-01-01T00:00:00Z",
            "2024-01-01T00:07:00Z",
            "2024-01-01T00:14:00Z",
        ]
    )
    df = pd.DataFrame({"timestamp": ts, "value": [1.0, 2.0, 3.0]})

    target_index = pd.date_range("2024-01-01", periods=6, freq="5min", tz="UTC")

    resampled = resample_left_label_last(
        df,
        timestamp_col="timestamp",
        freq="5min",
        columns=["value"],
        target_index=target_index,
    )

    assert list(resampled["timestamp"]) == list(target_index)
    assert str(resampled["timestamp"].dt.tz) == "UTC"

    expected = pd.Series([1.0, 1.0, 2.0, 3.0, 3.0, 3.0], index=target_index)
    expected.index.name = "timestamp"
    expected.name = "value"
    observed = resampled.set_index("timestamp")["value"]
    pd.testing.assert_series_equal(observed, expected, check_freq=False)


def test_assert_no_future_leak_rejects_future_rows() -> None:
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
        assert_no_future_leak(
            df,
            target_time_col="timestamp_target_open",
            feature_time_col="timestamp_feature",
        )

