"""Time handling utilities with strict UTC semantics."""

from __future__ import annotations

from typing import Iterable, Sequence

import pandas as pd

__all__ = [
    "ensure_utc_series",
    "ensure_utc_index",
    "resample_left_label_last",
    "assert_no_future_leak",
]


def _format_invalid_samples(values: Iterable[object], *, limit: int = 3) -> str:
    sample = list(values)[:limit]
    if not sample:
        return ""
    formatted = ", ".join(map(str, sample))
    if len(sample) == limit:
        return f" ({formatted}, ...)"
    return f" ({formatted})"


def _coerce_single_timestamp(value: object) -> pd.Timestamp | pd.NaT:
    if value is pd.NaT or pd.isna(value):
        return pd.NaT
    if isinstance(value, pd.Timestamp):
        return value.tz_localize("UTC") if value.tz is None else value.tz_convert("UTC")
    try:
        return pd.Timestamp(value, tz="UTC")
    except (TypeError, ValueError):  # pragma: no cover - defensive
        ts = pd.Timestamp(value)
        if ts.tz is None:
            return ts.tz_localize("UTC")
        return ts.tz_convert("UTC")


def ensure_utc_series(
    series: pd.Series,
    *,
    allow_na: bool = False,
    column_name: str | None = None,
) -> pd.Series:
    """Return a timezone-aware copy of *series* normalized to UTC."""

    if not isinstance(series, pd.Series):  # pragma: no cover - defensive guard
        raise TypeError("ensure_utc_series expects a pandas Series")

    converted = pd.to_datetime(series, utc=True, errors="coerce")
    if isinstance(converted, pd.Series):
        result = converted.copy()
    else:  # pragma: no cover - pandas guarantees Series in practice
        result = pd.Series(converted, index=series.index, name=series.name)

    original_not_na = series.notna()
    invalid = result.isna() & original_not_na
    if invalid.any():
        recovered = series.loc[invalid].apply(_coerce_single_timestamp)
        result.loc[invalid] = recovered.to_numpy()
        invalid = result.isna() & original_not_na

    if invalid.any() and not allow_na:
        bad_values = series.loc[invalid].tolist()
        hint = _format_invalid_samples(bad_values)
        col = column_name or series.name or "timestamp"
        raise ValueError(f"Column '{col}' contains non-parsable timestamps{hint}")

    return result


def ensure_utc_index(index: Sequence[pd.Timestamp] | pd.Index) -> pd.DatetimeIndex:
    """Normalize *index* to a :class:`~pandas.DatetimeIndex` with UTC tz."""

    idx = pd.Index(index)
    name = getattr(index, "name", idx.name)
    series = pd.Series(idx, name=name)
    coerced = ensure_utc_series(series, column_name=name or "timestamp")
    invalid = coerced.isna() & series.notna()
    if invalid.any():
        hint = _format_invalid_samples(series.loc[invalid].tolist())
        raise ValueError(f"Index contains non-parsable timestamps{hint}")
    return pd.DatetimeIndex(coerced.array, name=name)


def resample_left_label_last(
    df: pd.DataFrame,
    *,
    timestamp_col: str,
    freq: str,
    columns: Sequence[str] | None = None,
    target_index: Sequence[pd.Timestamp] | pd.Index | None = None,
    fill_method: str | None = "ffill",
    limit: int | None = None,
) -> pd.DataFrame:
    """Resample *df* using left-label semantics without introducing leakage."""

    if timestamp_col not in df.columns:
        raise KeyError(f"Column '{timestamp_col}' missing from dataframe")

    if columns is None:
        columns = [c for c in df.columns if c != timestamp_col]

    missing = [c for c in columns if c not in df.columns]
    if missing:
        joined = ", ".join(missing)
        raise KeyError(f"Columns {joined!r} missing from dataframe")

    working = df.loc[:, [timestamp_col, *columns]].copy()
    working[timestamp_col] = ensure_utc_series(
        working[timestamp_col], column_name=timestamp_col
    )
    working = working.sort_values(timestamp_col)
    indexed = working.set_index(timestamp_col)[columns]

    if target_index is None:
        base_index = indexed.index
        if base_index.empty:
            idx = base_index
        else:
            idx = pd.date_range(
                start=base_index.min(),
                end=base_index.max(),
                freq=freq,
                tz=base_index.tz,
            )
    else:
        idx = ensure_utc_index(target_index)

    combined_idx = idx.union(indexed.index)
    aligned = indexed.reindex(combined_idx).sort_index()

    if fill_method is None or fill_method == "none":
        filled = aligned
    elif fill_method == "ffill":
        filled = aligned.ffill(limit=limit)
    else:
        raise ValueError(f"Unsupported fill method: {fill_method!r}")

    trimmed = filled.reindex(idx)
    trimmed.index.name = timestamp_col
    result = trimmed.reset_index()
    return result


def assert_no_future_leak(
    df: pd.DataFrame,
    *,
    target_time_col: str,
    feature_time_col: str,
) -> None:
    """Assert that feature timestamps never peek into the future."""

    if target_time_col not in df.columns:
        raise KeyError(f"Column '{target_time_col}' missing from dataframe")
    if feature_time_col not in df.columns:
        raise KeyError(f"Column '{feature_time_col}' missing from dataframe")

    target_ts = ensure_utc_series(df[target_time_col], column_name=target_time_col)
    feature_ts = ensure_utc_series(
        df[feature_time_col], column_name=feature_time_col, allow_na=True
    )

    valid = feature_ts.notna()
    if not valid.any():
        return

    aligned_target = target_ts.loc[valid]
    aligned_feature = feature_ts.loc[valid]
    violations = aligned_feature > aligned_target
    if violations.any():
        offending = (
            pd.DataFrame(
                {
                    target_time_col: aligned_target.loc[violations],
                    feature_time_col: aligned_feature.loc[violations],
                }
            )
            .head()
            .to_string(index=False)
        )
        raise AssertionError(
            "Feature timestamps exceed their corresponding target timestamps. "
            f"Offending rows:\n{offending}"
        )

