"""Utility helpers for constructing derivative market features."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

__all__ = [
    "load_funding",
    "load_perp_basis",
    "load_open_interest",
    "make_deriv_features",
]


@dataclass(frozen=True)
class _LoaderConfig:
    column: str
    resample: str


def _ensure_timestamp_index(df: pd.DataFrame) -> pd.Series:
    if "timestamp" not in df.columns:
        raise KeyError("Dataframe must contain a 'timestamp' column")
    ts = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    if ts.isna().any():
        raise ValueError("timestamp column contains NaT values")
    return ts


def _infer_frequency(index: pd.DatetimeIndex) -> str:
    if index.freqstr:
        return str(index.freqstr)
    inferred = pd.infer_freq(index)
    if inferred is not None:
        return inferred
    diffs = index.to_series().diff().dropna()
    if diffs.empty:
        return "5T"
    return diffs.mode().iloc[0]


def _left_resample(series: pd.Series, freq: str, target_index: pd.DatetimeIndex | None) -> pd.Series:
    series = series.sort_index()
    resampled = series.resample(freq, label="left", closed="left").last().ffill()
    if target_index is not None:
        resampled = resampled.reindex(target_index, method="ffill")
    return resampled


def load_funding(
    df: pd.DataFrame,
    *,
    column: str = "funding_rate",
    freq: str | None = None,
    target_index: pd.DatetimeIndex | None = None,
) -> pd.Series:
    """Return a funding rate series aligned to the desired index."""

    ts = _ensure_timestamp_index(df)
    if column not in df.columns:
        raise KeyError(f"Column '{column}' missing from funding dataframe")
    series = pd.Series(df[column].to_numpy(dtype=float), index=ts)
    frequency = freq or _infer_frequency(series.index)
    return _left_resample(series, frequency, target_index)


def load_perp_basis(
    df: pd.DataFrame,
    *,
    column: str = "perp_basis",
    freq: str | None = None,
    target_index: pd.DatetimeIndex | None = None,
) -> pd.Series:
    """Return the perpetual basis aligned without look-ahead leakage."""

    ts = _ensure_timestamp_index(df)
    if column not in df.columns:
        raise KeyError(f"Column '{column}' missing from basis dataframe")
    series = pd.Series(df[column].to_numpy(dtype=float), index=ts)
    frequency = freq or _infer_frequency(series.index)
    return _left_resample(series, frequency, target_index)


def load_open_interest(
    df: pd.DataFrame,
    *,
    column: str = "open_interest",
    freq: str | None = None,
    target_index: pd.DatetimeIndex | None = None,
) -> pd.Series:
    """Return open interest aligned to the candle grid without leakage."""

    ts = _ensure_timestamp_index(df)
    if column not in df.columns:
        raise KeyError(f"Column '{column}' missing from open interest dataframe")
    series = pd.Series(df[column].to_numpy(dtype=float), index=ts)
    frequency = freq or _infer_frequency(series.index)
    return _left_resample(series, frequency, target_index)


def make_deriv_features(
    df: pd.DataFrame,
    *,
    funding_col: str = "funding_rate",
    basis_col: str = "perp_basis",
    oi_col: str = "open_interest",
    freq: str | None = None,
) -> pd.DataFrame:
    """Construct derivative features aligned to the provided price dataframe."""

    if "timestamp" not in df.columns:
        raise KeyError("Input dataframe must include 'timestamp'")

    timestamps = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    if timestamps.isna().any():
        raise ValueError("Input timestamps contain NaT values")

    base_index = pd.DatetimeIndex(timestamps).sort_values().unique()
    frequency = freq or _infer_frequency(base_index)

    features = pd.DataFrame(index=base_index)

    if funding_col in df.columns:
        funding_series = load_funding(
            df[["timestamp", funding_col]], freq=frequency, target_index=base_index
        )
        mean = float(funding_series.mean()) if not funding_series.empty else 0.0
        std = float(funding_series.std(ddof=0)) if not funding_series.empty else np.nan
        if std == 0 or np.isnan(std):
            funding_z = pd.Series(np.nan, index=funding_series.index)
        else:
            funding_z = (funding_series - mean) / std
        features["funding_z"] = funding_z.astype(np.float32)
    else:
        features["funding_z"] = np.nan

    if basis_col in df.columns:
        basis_series = load_perp_basis(
            df[["timestamp", basis_col]], freq=frequency, target_index=base_index
        )
        features["basis_bp"] = (basis_series * 10_000.0).astype(np.float32)
    else:
        features["basis_bp"] = np.nan

    if oi_col in df.columns:
        oi_series = load_open_interest(
            df[["timestamp", oi_col]], freq=frequency, target_index=base_index
        )
        oi_change = oi_series.pct_change().replace([np.inf, -np.inf], np.nan)
        features["oi_change_rate"] = oi_change.astype(np.float32)
    else:
        features["oi_change_rate"] = np.nan

    features = features.reset_index().rename(columns={"index": "timestamp"})
    return features
