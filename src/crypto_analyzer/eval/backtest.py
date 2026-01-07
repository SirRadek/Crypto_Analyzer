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
    fee_per_trade: float | None = None,
    fee_bps: float | None = None,
    prob_col: str | None = None,
    reward_col: str = "reward_ratio",
    risk_col: str = "risk_ratio",
    slippage_bps: float = 0.0,
    latency_steps: int = 0,
    p_touch_col: str | None = None,
    p_up_col: str | None = None,
    p_touch_threshold: float | None = None,
    p_up_threshold: float | None = None,
    position_size: float = 0.15,
    max_leverage: float = 3.0,
    max_trade_loss: float = 0.06,
    max_trade_gain: float = 0.30,
    compound: bool = False,
) -> BacktestResult:
    """Run a cost-aware backtest using an expected value decision rule."""

    if "timestamp" not in df.columns:
        raise KeyError("Input dataframe must include a 'timestamp' column")
    if "last_price" not in df.columns or "target" not in df.columns:
        raise KeyError("Dataframe must include 'last_price' and 'target' columns")

    ordered = df.sort_values("timestamp").reset_index(drop=True)
    working = ordered.copy()

    if latency_steps > 0:
        if latency_steps >= len(working):
            raise ValueError("Latency exceeds available data length")

        shift_cols = ["timestamp", "last_price", "target"]
        for extra in (reward_col, risk_col):
            if extra in working.columns and extra not in shift_cols:
                shift_cols.append(extra)

        for column in shift_cols:
            working[column] = working[column].shift(-latency_steps)

        working = working.iloc[:-latency_steps].reset_index(drop=True)

    price_return = (working["target"] - working["last_price"]) / working["last_price"]

    if fee_bps is not None and fee_per_trade is not None:
        raise ValueError("Specify either fee_per_trade or fee_bps, not both")
    if fee_bps is not None:
        fee_component = float(fee_bps / 10_000.0)
    else:
        fee_component = float(0.0004 if fee_per_trade is None else fee_per_trade)

    fee_total = float(fee_component + slippage_bps / 10_000.0)

    gating_mask = np.ones(len(working), dtype=bool)

    if p_touch_threshold is not None:
        touch_column = p_touch_col or prob_col
        if touch_column is None:
            raise ValueError("p_touch_threshold specified but no probability column provided")
        if touch_column not in working.columns:
            raise KeyError(f"Column '{touch_column}' required for p_touch_threshold")
        gating_mask &= working[touch_column].astype(np.float64).to_numpy() >= p_touch_threshold

    up_probs = None
    if p_up_col is not None:
        if p_up_col not in working.columns:
            raise KeyError(f"Column '{p_up_col}' required for p_up_threshold")
        up_probs = working[p_up_col].astype(np.float64).to_numpy()

    if prob_col is None:
        if "p_hat" not in working.columns:
            raise KeyError("Dataframe must include 'p_hat' when prob_col is None")
        if p_up_threshold is not None and up_probs is not None:
            direction = np.zeros(len(working), dtype=np.float64)
            direction[up_probs >= p_up_threshold] = 1.0
            direction[up_probs <= (1.0 - p_up_threshold)] = -1.0
        else:
            direction = np.where(working["p_hat"] > working["last_price"], 1.0, -1.0).astype(
                np.float64
            )

        direction = np.where(gating_mask, direction, 0.0)
        trade_ret = direction * price_return.to_numpy() - fee_total * np.abs(direction)
        ev = ((working["p_hat"] - working["last_price"]) / working["last_price"]).astype(
            np.float64
        ).to_numpy() - fee_total * np.abs(direction)
        trades = np.abs(direction) > 0
    else:
        probs = working[prob_col].astype(np.float64).clip(0.0, 1.0)
        prob_arr = probs.to_numpy()
        thr = p_up_threshold if p_up_threshold is not None else 0.5

        direction = np.zeros(len(working), dtype=np.float64)
        direction[prob_arr >= thr] = 1.0
        if thr > 0.5:
            direction[prob_arr <= (1.0 - thr)] = -1.0
        direction = np.where(gating_mask, direction, 0.0)

        ev = prob_arr - 0.5
        trade_ret = direction * price_return.to_numpy() - fee_total * np.abs(direction)
        trades = direction != 0

    exposure = float(np.clip(position_size, 0.0, 1.0)) * float(max(max_leverage, 0.0))
    trade_ret = trade_ret * exposure
    if max_trade_loss is not None or max_trade_gain is not None:
        lower = -float(max_trade_loss) if max_trade_loss is not None else None
        upper = float(max_trade_gain) if max_trade_gain is not None else None
        trade_ret = np.clip(
            trade_ret,
            lower if lower is not None else trade_ret.min(),
            upper if upper is not None else trade_ret.max(),
        )

    if compound:
        equity = (1.0 + trade_ret).cumprod()
    else:
        equity = 1.0 + np.cumsum(trade_ret)
    pnl = float(equity[-1] - 1.0)
    sharpe = float(np.mean(trade_ret) / (np.std(trade_ret) + 1e-9) * np.sqrt(len(trade_ret)))
    executed = np.abs(direction) > 0
    signed_returns = price_return.to_numpy() * np.sign(direction)
    hit_rate_vs_prediction = float(
        np.mean(signed_returns[executed] > 0) if np.any(executed) else float("nan")
    )
    hit_rate = hit_rate_vs_prediction
    avg_ev = float(np.mean(ev[executed])) if np.any(executed) else float("nan")

    running_max = np.maximum.accumulate(equity)
    drawdown = equity / running_max - 1.0
    max_drawdown = float(drawdown.min()) if len(drawdown) else 0.0

    metrics = {
        "pnl": pnl,
        "sharpe": sharpe,
        "trades": int(np.sum(trades)),
        "hit_rate": hit_rate,
        "avg_ev": avg_ev,
        "ev": avg_ev,
        "hit_rate_vs_prediction": hit_rate_vs_prediction,
        "max_drawdown": max_drawdown,
        "maxDD": max_drawdown,
    }
    equity_frame = pd.DataFrame({"timestamp": working["timestamp"], "equity": equity})
    return {"equity": equity_frame, "metrics": metrics}


__all__ = ["BacktestResult", "run_backtest"]
