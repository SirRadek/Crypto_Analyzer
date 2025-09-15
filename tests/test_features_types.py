import numpy as np
import pandas as pd

from analysis.feature_engineering import FEATURE_COLUMNS, create_features


def test_features_types():
    rng = np.random.default_rng(0)
    n = 100
    ts = pd.date_range("2024-01-01", periods=n, freq="5min", tz="UTC")
    close = 100 + rng.normal(scale=1, size=n).cumsum()
    high = close + rng.random(n)
    low = close - rng.random(n)
    volume = rng.random(n) + 1
    qvol = volume * close
    tbb = volume * 0.5
    tbq = qvol * 0.5
    df = pd.DataFrame(
        {
            "timestamp": ts,
            "close": close,
            "high": high,
            "low": low,
            "volume": volume,
            "quote_asset_volume": qvol,
            "taker_buy_base": tbb,
            "taker_buy_quote": tbq,
        }
    )
    feat_df = create_features(df)
    X = feat_df[FEATURE_COLUMNS]
    assert feat_df.columns.is_unique
    assert X.dtypes.eq("float32").all()
    assert not X.isna().any().any()
