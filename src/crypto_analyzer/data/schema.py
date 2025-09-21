"""Pandera schemas for validating market data inputs."""

from __future__ import annotations

import pandas as pd
import pandera as pa
from pandera import Check, Column, DataFrameSchema

from crypto_analyzer.utils.time import assert_no_future_leak, ensure_utc_series

__all__ = ["OHLCV_SCHEMA", "FEATURE_FRAME_SCHEMA"]


def _is_utc_series(series: pd.Series, *, allow_na: bool) -> bool:
    try:
        coerced = ensure_utc_series(series, allow_na=True)
    except ValueError:
        return False

    invalid = coerced.isna() & series.notna()
    if invalid.any():
        return False

    if not allow_na and coerced.isna().any():
        return False

    tz = coerced.dt.tz
    return tz is not None and str(tz) == "UTC"


def _timestamp_checks(monotonic: bool = True) -> list[Check]:
    checks = [Check(lambda s: _is_utc_series(s, allow_na=False), name="tz_utc")]
    if monotonic:
        checks.append(Check(lambda s: s.is_monotonic_increasing, name="monotonic_time"))
    return checks


def _left_label_check(df: pd.DataFrame) -> bool:
    try:
        assert_no_future_leak(
            df, target_time_col="timestamp_target_open", feature_time_col="timestamp_feature"
        )
    except AssertionError:
        return False
    return True


OHLCV_SCHEMA = DataFrameSchema(
    {
        "timestamp": Column(pa.DateTime, checks=_timestamp_checks(), nullable=False, coerce=True),
        "open": Column(pa.Float, nullable=False, coerce=True),
        "high": Column(pa.Float, nullable=False, coerce=True),
        "low": Column(pa.Float, nullable=False, coerce=True),
        "close": Column(pa.Float, nullable=False, coerce=True),
        "volume": Column(pa.Float, nullable=False, coerce=True),
        "quote_asset_volume": Column(pa.Float, nullable=False, coerce=True),
        "taker_buy_base": Column(pa.Float, nullable=False, coerce=True),
        "taker_buy_quote": Column(pa.Float, nullable=False, coerce=True),
        "number_of_trades": Column(pa.Int, nullable=True, coerce=True, required=False),
    },
    index=None,
    strict=False,
    coerce=True,
    unique=["timestamp"],
)


FEATURE_FRAME_SCHEMA = DataFrameSchema(
    {
        "timestamp_target_open": Column(
            pa.DateTime, checks=_timestamp_checks(monotonic=True), nullable=False, coerce=True
        ),
        "timestamp_feature": Column(
            pa.DateTime,
            checks=[Check(lambda s: _is_utc_series(s, allow_na=True), name="tz_utc")],
            nullable=True,
            coerce=True,
            required=True,
        ),
    },
    strict=False,
    coerce=True,
    checks=[Check(_left_label_check, name="left_label_no_leak")],
)

