from __future__ import annotations

import pandas as pd

from crypto_analyzer.eval.backtest import run_backtest


def _sample_predictions() -> pd.DataFrame:
    ts = pd.date_range("2024-01-01", periods=6, freq="5T", tz="UTC")
    base_price = 100.0
    prices = base_price + pd.Series([0, 1, 2, 3, 4, 5], dtype=float)
    targets = prices + pd.Series([0.5, 0.7, -0.3, 0.8, -0.2, 0.4], dtype=float)
    probs = pd.Series([0.7, 0.6, 0.4, 0.8, 0.45, 0.75], dtype=float)
    df = pd.DataFrame(
        {
            "timestamp": ts,
            "last_price": prices,
            "target": targets,
            "prob": probs,
            "reward_ratio": 0.01,
            "risk_ratio": 0.01,
        }
    )
    return df


def test_transaction_costs_reduce_pnl():
    df = _sample_predictions()
    base = run_backtest(df, fee_bps=0.0, slippage_bps=0.0, prob_col="prob")
    costly = run_backtest(df, fee_bps=50.0, slippage_bps=10.0, prob_col="prob")
    assert costly["metrics"]["pnl"] < base["metrics"]["pnl"]


def test_latency_shifts_candles():
    df = _sample_predictions()
    delayed = run_backtest(df, fee_bps=0.0, slippage_bps=0.0, prob_col="prob", latency_steps=2)
    assert len(delayed["equity"]) == len(df) - 2
