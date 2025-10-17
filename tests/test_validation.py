from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from crypto_analyzer.utils.config import CONFIG
from crypto_analyzer.utils.errors import DataValidationError
from crypto_analyzer.utils.validation import validate_features, validate_price_data


def _build_price_frame() -> pd.DataFrame:
    timestamps = pd.date_range(
        "2024-01-01", periods=4, freq=CONFIG.interval, tz="UTC"
    )
    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": np.linspace(10.0, 13.0, len(timestamps)),
            "high": np.linspace(11.0, 14.0, len(timestamps)),
            "low": np.linspace(9.0, 12.0, len(timestamps)),
            "close": np.linspace(10.5, 13.5, len(timestamps)),
            "volume": np.full(len(timestamps), 1000.0),
        }
    )


def test_validate_price_data_allows_regular_intervals() -> None:
    df = _build_price_frame()
    validate_price_data(df)


def test_validate_price_data_detects_missing_interval() -> None:
    df = _build_price_frame().iloc[:3].copy()
    df.loc[2, "timestamp"] = df.loc[1, "timestamp"] + 2 * pd.to_timedelta(CONFIG.interval)
    with pytest.raises(DataValidationError, match="missing or irregular interval"):
        validate_price_data(df)


def test_validate_price_data_detects_duplicate_timestamps() -> None:
    df = _build_price_frame().iloc[:3].copy()
    df.loc[2, "timestamp"] = df.loc[1, "timestamp"]
    with pytest.raises(DataValidationError, match="Duplicate timestamps"):
        validate_price_data(df)


def test_validate_price_data_detects_negative_values() -> None:
    df = _build_price_frame()
    df.loc[1, "close"] = -5.0
    with pytest.raises(DataValidationError, match="negative values"):
        validate_price_data(df)


def test_validate_features_accepts_finite_values() -> None:
    df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=3, freq="D", tz="UTC"),
            "feature_a": [0.0, -1.0, 2.0],
            "feature_b": [0.5, 0.75, 1.0],
        }
    )
    validate_features(df)


def test_validate_features_detects_nan() -> None:
    df = pd.DataFrame({"feature_a": [0.0, np.nan]})
    with pytest.raises(DataValidationError, match="Feature column 'feature_a'"):
        validate_features(df)


def test_validate_features_detects_infinite_values() -> None:
    df = pd.DataFrame({"feature_a": [0.0, np.inf]})
    with pytest.raises(DataValidationError, match="Feature column 'feature_a'"):
        validate_features(df)
