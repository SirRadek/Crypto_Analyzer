import numpy as np
import pandas as pd

from analysis.feature_engineering import create_features


def test_future_extreme_targets_once():
    n = 60
    ts = pd.date_range("2024-01-01", periods=n, freq="15min", tz="UTC")
    close = np.linspace(100, 101, n)
    high = close + 1
    low = close - 1
    open_ = (high + low) / 2
    volume = np.ones(n)
    qvol = volume * close
    tbb = volume * 0.5
    tbq = qvol * 0.5
    df = pd.DataFrame(
        {
            "timestamp": ts,
            "open": open_,
            "close": close,
            "high": high,
            "low": low,
            "volume": volume,
            "quote_asset_volume": qvol,
            "taker_buy_base": tbb,
            "taker_buy_quote": tbq,
        }
    )

    out = create_features(df)
    cols = [c for c in out.columns if c.startswith("delta_low") or c.startswith("delta_high")]
    assert cols == [
        "delta_low_log_120m",
        "delta_low_lin_120m",
        "delta_high_log_120m",
        "delta_high_lin_120m",
        "delta_low_log_60m",
        "delta_low_lin_60m",
        "delta_high_log_60m",
        "delta_high_lin_60m",
        "delta_low_log_240m",
        "delta_low_lin_240m",
        "delta_high_log_240m",
        "delta_high_lin_240m",
    ]
    for c in cols:
        assert out[c].dtype == np.float32
