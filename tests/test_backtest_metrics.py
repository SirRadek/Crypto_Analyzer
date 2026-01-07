import numpy as np
import pandas as pd
import pytest

from crypto_analyzer.eval.backtest import run_backtest


def test_run_backtest_produces_equity_and_metrics():
    ts = pd.date_range("2024-01-01", periods=20, freq="1D", tz="UTC")
    last_price = np.linspace(100.0, 102.0, num=len(ts), dtype=np.float32)
    target = last_price + 0.5
    preds = last_price + np.linspace(0.1, 0.6, num=len(ts), dtype=np.float32)

    df = pd.DataFrame(
        {
            "timestamp": ts,
            "p_hat": preds,
            "target": target,
            "last_price": last_price,
        }
    )

    result = run_backtest(df, fee_per_trade=0.0)
    equity = result["equity"]
    metrics = result["metrics"]

    assert list(equity.columns) == ["timestamp", "equity"]
    assert equity["equity"].iloc[0] == pytest.approx(1.0, rel=5e-3)
    assert equity["equity"].iloc[-1] >= equity["equity"].iloc[0]
    expected_keys = {
        "pnl",
        "sharpe",
        "trades",
        "hit_rate",
        "avg_ev",
        "ev",
        "hit_rate_vs_prediction",
        "max_drawdown",
    }
    assert expected_keys.issubset(metrics.keys())
    assert metrics["pnl"] >= 0.0
    assert np.isfinite(metrics["sharpe"])
    assert metrics["ev"] == pytest.approx(metrics["avg_ev"])
    assert metrics["hit_rate"] == pytest.approx(metrics["hit_rate_vs_prediction"])
    assert metrics["max_drawdown"] <= 0.0
