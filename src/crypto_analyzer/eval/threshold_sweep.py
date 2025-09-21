"""Utilities for sweeping backtest probability thresholds."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from .backtest import run_backtest


@dataclass(frozen=True)
class SweepParams:
    """Configuration for a threshold sweep."""

    p_touch_values: Sequence[float]
    p_up_values: Sequence[float]
    fee_bps: float
    slippage_bps: float
    latency_steps: int
    p_touch_col: str = "p_hat"
    p_up_col: str | None = None


def _ensure_iterable(values: Sequence[float] | Iterable[float]) -> list[float]:
    arr = list(values)
    if not arr:
        raise ValueError("Threshold grid must contain at least one value")
    return [float(v) for v in arr]


def sweep_threshold_grid(
    df: pd.DataFrame,
    *,
    params: SweepParams,
    horizon_label: str | int | float | None = None,
) -> pd.DataFrame:
    """Run a grid search over EV thresholds for a single horizon.

    Parameters
    ----------
    df:
        Normalised prediction dataframe containing ``timestamp``, ``last_price``
        and ``target`` columns alongside probability estimates.
    params:
        Sweep configuration specifying the grids and execution costs.
    horizon_label:
        Optional label identifying the horizon these predictions belong to.
    """

    required = {"timestamp", "last_price", "target"}
    missing = required.difference(df.columns)
    if missing:
        raise KeyError(f"Input dataframe is missing columns: {sorted(missing)!r}")

    p_touch_values = _ensure_iterable(params.p_touch_values)
    p_up_values = _ensure_iterable(params.p_up_values)

    if params.p_up_col is None or params.p_up_col not in df.columns:
        p_up_values = [np.nan]
        p_up_col = None
    else:
        p_up_col = params.p_up_col

    records: list[dict[str, float | int | str | None]] = []

    working = df.sort_values("timestamp").reset_index(drop=True)

    for p_touch in p_touch_values:
        for p_up in p_up_values:
            result = run_backtest(
                working,
                fee_bps=params.fee_bps,
                slippage_bps=params.slippage_bps,
                latency_steps=max(0, int(params.latency_steps)),
                p_touch_col=params.p_touch_col,
                p_touch_threshold=float(p_touch),
                p_up_col=p_up_col,
                p_up_threshold=float(p_up) if np.isfinite(p_up) else None,
            )
            metrics = result["metrics"]
            records.append(
                {
                    "horizon": horizon_label,
                    "p_touch_thr": float(p_touch),
                    "p_up_thr": float(p_up) if np.isfinite(p_up) else np.nan,
                    "ev": float(metrics.get("ev", np.nan)),
                    "pnl": float(metrics.get("pnl", np.nan)),
                    "sharpe": float(metrics.get("sharpe", np.nan)),
                    "hit_rate": float(metrics.get("hit_rate", np.nan)),
                    "trades": float(metrics.get("trades", np.nan)),
                }
            )

    return pd.DataFrame.from_records(records)


__all__ = ["SweepParams", "sweep_threshold_grid"]
