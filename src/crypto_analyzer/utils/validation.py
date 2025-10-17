"""Data validation helpers for pipeline processing."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

from crypto_analyzer.utils.config import CONFIG
from crypto_analyzer.utils.errors import DataValidationError

__all__ = ["validate_price_data", "validate_features"]


def _ensure_required_columns(df: pd.DataFrame, columns: Iterable[str]) -> None:
    missing = [col for col in columns if col not in df.columns]
    if missing:
        joined = ", ".join(sorted(missing))
        raise DataValidationError(f"Missing required column(s): {joined}.")


def validate_price_data(df: pd.DataFrame) -> None:
    """Validate continuity and integrity of raw OHLCV price data.

    Parameters
    ----------
    df:
        Data frame expected to contain at least ``timestamp`` and OHLCV columns.

    Raises
    ------
    DataValidationError
        If timestamps are not strictly increasing, intervals are missing, or
        if any price/volume values are negative.
    """

    if df.empty:
        raise DataValidationError("Price data is empty.")

    _ensure_required_columns(df, ["timestamp"])

    try:
        timestamps = pd.to_datetime(df["timestamp"], utc=True)
    except (TypeError, ValueError) as exc:
        raise DataValidationError("Unable to parse timestamps in price data.") from exc

    if timestamps.isna().any():
        raise DataValidationError("Price data contains invalid or missing timestamps.")

    if not timestamps.is_monotonic_increasing:
        raise DataValidationError("Timestamps must be strictly increasing.")

    if not timestamps.is_unique:
        raise DataValidationError("Duplicate timestamps detected in price data.")

    try:
        expected_delta = pd.to_timedelta(CONFIG.interval)
    except ValueError as exc:
        raise DataValidationError("Invalid interval configuration for validation.") from exc

    for idx in range(1, len(timestamps)):
        delta = timestamps.iloc[idx] - timestamps.iloc[idx - 1]
        if delta != expected_delta:
            raise DataValidationError(
                "Detected missing or irregular interval between "
                f"{timestamps.iloc[idx - 1]} and {timestamps.iloc[idx]}."
            )

    numeric_columns = [
        col for col in ("open", "high", "low", "close", "volume") if col in df.columns
    ]
    if numeric_columns:
        for column in numeric_columns:
            values = pd.to_numeric(df[column], errors="coerce")
            if np.isnan(values).any():
                raise DataValidationError(
                    f"Column '{column}' contains non-numeric or missing values in price data."
                )
            if not np.isfinite(values).all():
                raise DataValidationError(
                    f"Column '{column}' contains non-finite values in price data."
                )
            if (values < 0).any():
                raise DataValidationError(
                    f"Column '{column}' contains negative values in price data."
                )


def validate_features(df: pd.DataFrame) -> None:
    """Validate engineered features for downstream model consumption."""

    if df.empty:
        raise DataValidationError("Feature data is empty after engineering.")

    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.empty:
        return

    invalid_mask = ~np.isfinite(numeric_df.values)
    if invalid_mask.any():
        bad_row, bad_col = np.argwhere(invalid_mask)[0]
        column_name = numeric_df.columns[bad_col]
        raise DataValidationError(
            f"Feature column '{column_name}' contains NaN or infinite values."
        )
