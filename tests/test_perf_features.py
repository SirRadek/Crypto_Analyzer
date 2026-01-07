import os
import time

import numpy as np
import pandas as pd
import pytest

from crypto_analyzer.features.engineering import create_features
from crypto_analyzer.utils.config import FeatureSettings


def _synthetic_prices(rows: int) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    ts = pd.date_range("2024-01-01", periods=rows, freq="1D", tz="UTC")
    close = 100 + rng.normal(scale=1.0, size=rows).cumsum()
    open_ = close - rng.random(rows)
    high = close + rng.random(rows)
    low = close - rng.random(rows)
    volume = rng.random(rows) + 1.0
    qvol = volume * close
    tbb = volume * 0.5
    tbq = qvol * 0.5
    return pd.DataFrame(
        {
            "timestamp": ts,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
            "quote_asset_volume": qvol,
            "taker_buy_base": tbb,
            "taker_buy_quote": tbq,
        }
    )


def test_create_features_runtime_budget() -> None:
    budget = float(os.getenv("PERF_FEATURE_BUDGET_SEC", "2.5"))
    if budget <= 0:
        pytest.skip("PERF_FEATURE_BUDGET_SEC disabled")

    settings = FeatureSettings(
        include_onchain=False,
        include_orderbook=False,
        include_derivatives=False,
        include_sentiment=False,
        forward_fill_limit=0,
        fillna_value=0.0,
    )
    df = _synthetic_prices(500)

    start = time.perf_counter()
    create_features(df, settings=settings)
    elapsed = time.perf_counter() - start

    assert elapsed <= budget, f"create_features took {elapsed:.3f}s > {budget:.3f}s budget"
