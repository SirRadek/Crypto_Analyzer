"""Helpers for volatility regime tagging and metric aggregation."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd


def assign_volatility_regimes(
    df: pd.DataFrame,
    *,
    price_col: str = "close",
    window: int = 96,
    calm_quantile: float = 0.3,
    volatile_quantile: float = 0.7,
) -> pd.Series:
    """Tag each observation with a calm/neutral/volatile regime label."""

    if price_col not in df.columns:
        raise KeyError(f"Column {price_col!r} not found in dataframe")
    if not 0.0 < calm_quantile < volatile_quantile < 1.0:
        raise ValueError("Quantiles must satisfy 0 < calm < volatile < 1")

    prices = df[price_col].astype(np.float64)
    returns = prices.pct_change().fillna(0.0)
    realized_vol = returns.rolling(window, min_periods=window // 2).std().bfill()

    q_low = realized_vol.quantile(calm_quantile)
    q_high = realized_vol.quantile(volatile_quantile)

    regime = np.where(
        realized_vol <= q_low,
        "calm",
        np.where(realized_vol >= q_high, "volatile", "neutral"),
    )
    return pd.Series(regime, index=df.index, dtype="string")


def metric_by_regime(
    y_true: pd.Series | np.ndarray,
    y_pred: pd.Series | np.ndarray,
    regimes: pd.Series | np.ndarray,
    *,
    metric: Callable[[np.ndarray, np.ndarray], float],
) -> pd.DataFrame:
    """Aggregate ``metric`` separately for each supplied regime label."""

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    regimes = pd.Series(regimes, dtype="string")
    if not (len(y_true) == len(y_pred) == len(regimes)):
        raise ValueError("All inputs must have matching lengths")

    records: list[dict[str, object]] = []
    for label in regimes.dropna().unique():
        mask = regimes == label
        if mask.sum() == 0:
            continue
        score = metric(y_true[mask], y_pred[mask])
        records.append({"regime": str(label), "score": float(score), "count": int(mask.sum())})

    return pd.DataFrame.from_records(records)


__all__ = ["assign_volatility_regimes", "metric_by_regime"]
