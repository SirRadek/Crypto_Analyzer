"""Backtesting utilities with cost-aware decision rules."""

from __future__ import annotations

from typing import TypedDict

import numpy as np
import pandas as pd


class BacktestResult(TypedDict):
    equity: pd.DataFrame
    metrics: dict[str, float]


def run_backtest(
    df: pd.DataFrame,
    *,
    fee_per_trade: float = 0.0004,
    prob_col: str | None = None,
    reward_col: str = "reward_ratio",
    risk_col: str = "risk_ratio",
    slippage_bps: float = 0.0,
    latency_steps: int = 0,
) -> BacktestResult:
    """Run a cost-aware backtest using an expected value decision rule."""

    if "timestamp" not in df.columns:
        raise KeyError("Input dataframe must include a 'timestamp' column")
    if "last_price" not in df.columns or "target" not in df.columns:
        raise KeyError("Dataframe must include 'last_price' and 'target' columns")

    price_return = (df["target"] - df["last_price"]) / df["last_price"]
    fee_total = float(fee_per_trade + slippage_bps / 10_000.0)

    if prob_col is None:
        if "p_hat" not in df.columns:
            raise KeyError("Dataframe must include 'p_hat' when prob_col is None")
        direction = np.where(df["p_hat"] > df["last_price"], 1.0, -1.0)
        trade_ret = direction * price_return.to_numpy() - fee_total * np.abs(direction)
        ev = np.zeros_like(trade_ret)
        trades = np.abs(direction) > 0
    else:
        probs = df[prob_col].astype(np.float64)
        if latency_steps > 0:
            probs = probs.shift(latency_steps)
        probs = probs.fillna(0.0).clip(0.0, 1.0)

        reward_series = (
            df[reward_col]
            if reward_col in df.columns
            else price_return.clip(lower=0.0)
        ).astype(np.float64).to_numpy()
        risk_series = (
            df[risk_col]
            if risk_col in df.columns
            else (-price_return).clip(lower=0.0)
        ).astype(np.float64).to_numpy()

        prob_arr = probs.to_numpy()
        ev = prob_arr * reward_series - (1.0 - prob_arr) * risk_series - fee_total
        direction = np.where(ev > 0.0, 1.0, 0.0)
        trade_ret = direction * price_return.to_numpy() - fee_total * direction
        trades = direction > 0

    equity = (1.0 + trade_ret).cumprod()
    pnl = float(equity[-1] - 1.0)
    sharpe = float(np.mean(trade_ret) / (np.std(trade_ret) + 1e-9) * np.sqrt(len(trade_ret)))
    hit_rate = float(
        np.mean(price_return.to_numpy()[trades] > 0) if np.any(trades) else np.nan
    )
    avg_ev = float(np.mean(ev[trades])) if np.any(trades) else float("nan")

    metrics = {
        "pnl": pnl,
        "sharpe": sharpe,
        "trades": int(np.sum(trades)),
        "hit_rate": hit_rate,
        "avg_ev": avg_ev,
    }
    equity_frame = pd.DataFrame({"timestamp": df["timestamp"], "equity": equity})
    return {"equity": equity_frame, "metrics": metrics}


__all__ = ["BacktestResult", "run_backtest"]
