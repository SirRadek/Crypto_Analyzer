"""Utility helpers for constructing derivative market features."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from crypto_analyzer.utils.config import CONFIG, DerivativeDataSettings

__all__ = [
    "load_funding",
    "load_perp_basis",
    "load_open_interest",
    "make_deriv_features",
]

_FUNDING_ALIASES: tuple[str, ...] = ("deriv_funding_rate", "funding_8h")
_BASIS_ALIASES: tuple[str, ...] = ("basis_annualized", "basis", "perp_basis")
_OI_ALIASES: tuple[str, ...] = ("oi", "openinterest")


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
    if len(index) <= 1:
        return "5T"
    diffs = index.sort_values().to_series().diff().dropna()
    if diffs.empty:
        return "5T"
    return diffs.mode().iloc[0]


def _left_resample(
    series: pd.Series,
    freq: str,
    target_index: pd.DatetimeIndex | None,
) -> pd.Series:
    series = series.sort_index()
    resampled = series.resample(freq, label="left", closed="left").last().ffill()
    if target_index is not None:
        resampled = resampled.reindex(target_index, method="ffill")
    return resampled


def _read_source(
    source: pd.DataFrame | str | Path | None,
) -> pd.DataFrame | None:
    if source is None:
        return None
    if isinstance(source, pd.DataFrame):
        return source.copy()
    path = Path(source)
    if not path.is_file():
        raise FileNotFoundError(f"Derivative data file not found: {path}")
    if path.suffix.lower() in {".parquet", ".pq"}:
        frame = pd.read_parquet(path)
    else:
        frame = pd.read_csv(path)
    return frame


def _find_column(columns: Iterable[str], primary: str, aliases: Iterable[str]) -> str | None:
    if primary in columns:
        return primary
    for candidate in aliases:
        if candidate in columns:
            return candidate
    return None


def _prepare_source(
    df: pd.DataFrame,
    column: str,
    *,
    source: pd.DataFrame | str | Path | None,
    aliases: Iterable[str] = (),
) -> pd.DataFrame | None:
    if column in df.columns:
        return df.loc[:, ["timestamp", column]].copy()

    external = _read_source(source)
    if external is None or external.empty:
        return None

    actual = _find_column(external.columns, column, aliases)
    if actual is None:
        return None

    subset = external.loc[:, ["timestamp", actual]].copy()
    if actual != column:
        subset = subset.rename(columns={actual: column})
    return subset


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


def _maybe_load_series(
    df: pd.DataFrame,
    column: str,
    loader,
    *,
    source: pd.DataFrame | str | Path | None,
    aliases: Iterable[str],
    freq: str,
    target_index: pd.DatetimeIndex,
) -> pd.Series:
    frame = _prepare_source(df, column, source=source, aliases=aliases)
    if frame is None or frame.empty:
        return pd.Series(np.nan, index=target_index, dtype=float)
    try:
        return loader(frame, column=column, freq=freq, target_index=target_index)
    except (KeyError, ValueError):  # pragma: no cover - defensive
        return pd.Series(np.nan, index=target_index, dtype=float)


def _resolve_derivative_settings(
    config: DerivativeDataSettings | None,
) -> DerivativeDataSettings | None:
    if config is not None:
        return config
    try:
        return CONFIG.derivatives
    except AttributeError:  # pragma: no cover - defensive
        return None


def make_deriv_features(
    df: pd.DataFrame,
    *,
    funding_col: str = "funding_rate",
    basis_col: str = "perp_basis",
    oi_col: str = "open_interest",
    freq: str | None = None,
    funding_source: pd.DataFrame | str | Path | None = None,
    basis_source: pd.DataFrame | str | Path | None = None,
    oi_source: pd.DataFrame | str | Path | None = None,
    config: DerivativeDataSettings | None = None,
) -> pd.DataFrame:
    """Construct derivative features aligned to the provided price dataframe."""

    if "timestamp" not in df.columns:
        raise KeyError("Input dataframe must include 'timestamp'")

    timestamps = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    if timestamps.isna().any():
        raise ValueError("Input timestamps contain NaT values")

    base_index = pd.DatetimeIndex(timestamps).sort_values().unique()
    cfg = _resolve_derivative_settings(config)
    cfg_freq = cfg.resample_freq if cfg is not None else None
    frequency = freq or cfg_freq or _infer_frequency(base_index)

    if cfg is not None:
        funding_source = funding_source or cfg.funding_source
        basis_source = basis_source or cfg.basis_source
        oi_source = oi_source or cfg.open_interest_source

    features = pd.DataFrame(index=base_index)

    funding_series = _maybe_load_series(
        df,
        funding_col,
        load_funding,
        source=funding_source,
        aliases=_FUNDING_ALIASES,
        freq=frequency,
        target_index=base_index,
    )
    if funding_series.notna().any():
        mean = float(funding_series.mean())
        std = float(funding_series.std(ddof=0))
        if std == 0 or np.isnan(std):
            funding_z = pd.Series(np.nan, index=funding_series.index)
        else:
            funding_z = (funding_series - mean) / std
    else:
        funding_z = pd.Series(np.nan, index=funding_series.index)
    features["funding_z"] = funding_z.astype(np.float32)

    basis_series = _maybe_load_series(
        df,
        basis_col,
        load_perp_basis,
        source=basis_source,
        aliases=_BASIS_ALIASES,
        freq=frequency,
        target_index=base_index,
    )
    features["basis_bp"] = (basis_series * 10_000.0).astype(np.float32)

    oi_series = _maybe_load_series(
        df,
        oi_col,
        load_open_interest,
        source=oi_source,
        aliases=_OI_ALIASES,
        freq=frequency,
        target_index=base_index,
    )
    oi_change = oi_series.pct_change().replace([np.inf, -np.inf], np.nan)
    features["oi_change_rate"] = oi_change.astype(np.float32)

    features = features.reset_index().rename(columns={"index": "timestamp"})
    return features
