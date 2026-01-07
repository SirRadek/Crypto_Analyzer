import numpy as np
import pandas as pd
import pytest

from crypto_analyzer.eval.backtest import run_backtest


FEE_PER_TRADE = 0.001
SLIPPAGE_BPS = 5
TOTAL_COST = FEE_PER_TRADE + SLIPPAGE_BPS / 10_000


def _manual_backtest(
    df: pd.DataFrame, latency_steps: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    working = df.sort_values("timestamp").reset_index(drop=True)

    if latency_steps > 0:
        shift_cols = ["timestamp", "last_price", "target", "reward", "risk"]
        for column in shift_cols:
            working[column] = working[column].shift(-latency_steps)
        working = working.iloc[:-latency_steps].reset_index(drop=True)

    price_return = (working["target"] - working["last_price"]) / working["last_price"]
    prob_arr = working["prob"].to_numpy()
    reward = working["reward"].to_numpy()
    risk = working["risk"].to_numpy()

    ev = prob_arr * reward - (1.0 - prob_arr) * risk - TOTAL_COST
    direction = (ev > 0.0).astype(float)
    trade_ret = direction * price_return.to_numpy() - TOTAL_COST * direction
    equity = np.cumprod(1.0 + trade_ret)

    return trade_ret, equity, direction


def test_backtest_applies_costs_and_latency():
    df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=4, freq="h", tz="UTC"),
            "last_price": [100.0, 100.0, 100.0, 100.0],
            "target": [102.0, 99.0, 101.5, 103.0],
            "prob": [0.7, 0.55, 0.4, 0.8],
            "reward": [0.03, 0.025, 0.02, 0.035],
            "risk": [0.01, 0.02, 0.015, 0.02],
        }
    )

    result = run_backtest(
        df,
        prob_col="prob",
        reward_col="reward",
        risk_col="risk",
        fee_per_trade=FEE_PER_TRADE,
        slippage_bps=SLIPPAGE_BPS,
    )

    _, manual_equity, manual_direction = _manual_backtest(df, latency_steps=0)

    np.testing.assert_allclose(result["equity"]["equity"].to_numpy(), manual_equity)
    assert result["metrics"]["trades"] == int(manual_direction.sum())
    assert result["metrics"]["pnl"] == pytest.approx(float(manual_equity[-1] - 1.0))

    latency_result = run_backtest(
        df,
        prob_col="prob",
        reward_col="reward",
        risk_col="risk",
        fee_per_trade=FEE_PER_TRADE,
        slippage_bps=SLIPPAGE_BPS,
        latency_steps=1,
    )

    _, manual_equity_lat, manual_direction_lat = _manual_backtest(df, latency_steps=1)

    np.testing.assert_allclose(latency_result["equity"]["equity"].to_numpy(), manual_equity_lat)
    assert latency_result["metrics"]["trades"] == int(manual_direction_lat.sum())
    assert latency_result["equity"]["timestamp"].iloc[0] == df["timestamp"].iloc[1]
    assert latency_result["metrics"]["pnl"] == pytest.approx(float(manual_equity_lat[-1] - 1.0))
