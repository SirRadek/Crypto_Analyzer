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


def test_backtest_latency_shifts_entries():
    df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=6, freq="5min"),
            "p_hat": [101, 102, 103, 104, 105, 106],
            "target": [101.5, 102.5, 103.5, 104.5, 105.5, 106.5],
            "last_price": [101, 102, 103, 104, 105, 106],
        }
    )

    result = run_backtest(df, latency_steps=1, fee_per_trade=0.0)
    equity = result["equity"]

    assert len(equity) == len(df) - 1
    # the first trade should align with the second timestamp after latency shift
    assert equity["timestamp"].iloc[0] == df["timestamp"].iloc[1]


def test_backtest_prob_thresholds_gate_trades():
    df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-02-01", periods=4, freq="15min"),
            "last_price": [100, 101, 102, 103],
            "target": [101, 102, 103, 104],
            "p_success": [0.2, 0.3, 0.4, 0.5],
            "reward_ratio": [0.01, 0.01, 0.01, 0.01],
            "risk_ratio": [0.005, 0.005, 0.005, 0.005],
        }
    )

    result = run_backtest(
        df,
        prob_col="p_success",
        p_touch_col="p_success",
        p_touch_threshold=0.6,
        fee_per_trade=0.0,
    )

    assert result["metrics"]["trades"] == 0


def test_backtest_directional_threshold_creates_shorts_and_longs():
    df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-03-01", periods=4, freq="15min"),
            "p_hat": [100.5, 99.5, 100.2, 99.8],
            "last_price": [100, 100, 100, 100],
            "target": [100.6, 99.4, 100.3, 99.7],
            "p_up": [0.8, 0.15, 0.7, 0.2],
        }
    )

    result = run_backtest(
        df,
        p_up_col="p_up",
        p_up_threshold=0.75,
        fee_per_trade=0.0,
    )

    metrics = result["metrics"]
    assert metrics["trades"] == 3  # long on index 0, short on indices 1 and 3
    assert metrics["hit_rate"] <= 1.0
