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
