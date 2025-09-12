import pandas as pd

from ml.backtest import run_backtest


def test_backtest_equity_length():
    df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024", periods=5, freq="5min"),
            "p_low": [90, 91, 92, 93, 94],
            "p_hat": [101, 102, 103, 104, 105],
            "p_high": [110, 111, 112, 113, 114],
            "target": [100, 101, 102, 103, 104],
            "last_price": [100, 101, 102, 103, 104],
        }
    )
    res = run_backtest(df)
    assert len(res["equity"]) == len(df)
