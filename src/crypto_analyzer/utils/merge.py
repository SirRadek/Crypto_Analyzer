"""Utility helpers for safely merging time-aligned datasets."""

from __future__ import annotations

from typing import Sequence

import pandas as pd

from crypto_analyzer.utils.time import assert_no_future_leak, ensure_utc_series

__all__ = ["merge_left_labeled", "validate_left_label_alignment"]


def validate_left_label_alignment(
    df: pd.DataFrame,
    *,
    target_ts_col: str = "timestamp_target_open",
    feature_ts_col: str = "timestamp_feature",
) -> None:
    """Assert that *feature_ts_col* never looks into the future relative to the target."""

    if target_ts_col not in df.columns:
        raise KeyError(f"Column '{target_ts_col}' missing from dataframe")
    if feature_ts_col not in df.columns:
        raise KeyError(f"Column '{feature_ts_col}' missing from dataframe")

    assert_no_future_leak(
        df, target_time_col=target_ts_col, feature_time_col=feature_ts_col
    )


def merge_left_labeled(
    targets: pd.DataFrame,
    features: pd.DataFrame,
    *,
    target_ts_col: str = "timestamp_target_open",
    feature_ts_col: str = "timestamp_feature",
    feature_columns: Sequence[str] | None = None,
    suffixes: tuple[str, str] = ("", "_feature"),
) -> pd.DataFrame:
    """Merge ``targets`` with ``features`` using left-label resampling semantics."""

    if target_ts_col not in targets.columns:
        raise KeyError(f"Column '{target_ts_col}' missing from targets dataframe")
    if feature_ts_col not in features.columns:
        raise KeyError(f"Column '{feature_ts_col}' missing from features dataframe")

    left = targets.copy()
    left["__orig_order"] = range(len(left))
    left[target_ts_col] = ensure_utc_series(left[target_ts_col], column_name=target_ts_col)
    left = left.sort_values(target_ts_col).reset_index(drop=True)

    right = features.copy()
    right[feature_ts_col] = ensure_utc_series(
        right[feature_ts_col], column_name=feature_ts_col, allow_na=True
    )
    right = right.dropna(subset=[feature_ts_col]).sort_values(feature_ts_col).reset_index(drop=True)

    if feature_columns is None:
        merge_cols: list[str] = [c for c in right.columns if c != feature_ts_col]
    else:
        missing = [c for c in feature_columns if c not in right.columns]
        if missing:
            raise KeyError(f"Columns {missing!r} missing from features dataframe")
        merge_cols = feature_columns

    right_merge = right[[feature_ts_col, *merge_cols]].copy()

    merged = pd.merge_asof(
        left,
        right_merge,
        left_on=target_ts_col,
        right_on=feature_ts_col,
        direction="backward",
        allow_exact_matches=True,
        suffixes=suffixes,
    )

    merged = merged.sort_values("__orig_order").drop(columns="__orig_order").reset_index(drop=True)

    validate_left_label_alignment(
        merged, target_ts_col=target_ts_col, feature_ts_col=feature_ts_col
    )

    return merged

