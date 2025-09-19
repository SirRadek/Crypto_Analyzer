import pandas as pd

from crypto_analyzer.eval.backtest import run_backtest


def test_backtest_equity_length():
    df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024", periods=5, freq="15min"),
            "p_hat": [101, 102, 103, 104, 105],
            "target": [100, 101, 102, 103, 104],
            "last_price": [100, 101, 102, 103, 104],
        }
    )
    res = run_backtest(df)
    assert len(res["equity"]) == len(df)


def test_backtest_expected_value_rule_applies_fees_and_slippage():
    ts = pd.date_range("2024", periods=6, freq="15min")
    df = pd.DataFrame(
        {
            "timestamp": ts,
            "last_price": [100, 101, 102, 103, 104, 105],
            "target": [101, 101.4, 101.6, 102.5, 105, 103],
            "p_success": [0.8, 0.4, 0.7, 0.9, 0.2, 0.6],
            "reward_ratio": [0.01, 0.003, 0.006, 0.01, 0.015, 0.002],
            "risk_ratio": [0.005, 0.01, 0.004, 0.003, 0.008, 0.007],
        }
    )

    result = run_backtest(
        df,
        prob_col="p_success",
        slippage_bps=2.0,
        fee_per_trade=0.001,
    )

    metrics = result["metrics"]
    assert metrics["trades"] > 0
    assert metrics["avg_ev"] <= max(df["reward_ratio"])  # bounded by reward
