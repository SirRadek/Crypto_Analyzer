import numpy as np
import pandas as pd

import ml.time_cv as time_cv
import ml.train_price as tp
from analysis.feature_engineering import FEATURE_COLUMNS


def test_smoke_train_price(monkeypatch, tmp_path):
    rng = np.random.default_rng(0)
    n = 1050
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

    def small_reg():
        params = {
            "max_depth": 2,
            "eta": 0.1,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "tree_method": "hist",
            "eval_metric": "rmse",
            "nthread": 1,
            "seed": 42,
        }
        return params, 5

    def small_bound(kind: str):
        params = {
            "max_depth": 2,
            "eta": 0.1,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "tree_method": "hist",
            "eval_metric": "rmse",
            "nthread": 1,
            "seed": 42,
        }
        return params, 5

    monkeypatch.setattr(tp, "build_reg", small_reg)
    monkeypatch.setattr(tp, "build_bound", small_bound)

    def small_folds(n_samples, embargo=24):
        return time_cv.time_folds(n_samples, n_splits=2, embargo=embargo)

    monkeypatch.setattr(tp, "time_folds", small_folds)

    metrics, preds = tp.train_price(df, FEATURE_COLUMNS, outdir=tmp_path)
    assert ((preds["p_low"] <= preds["p_hat"]) & (preds["p_hat"] <= preds["p_high"])).all()
    assert (tmp_path / "low.joblib").exists()
    assert (tmp_path / "high.joblib").exists()
    cov = ((preds["target"] >= preds["p_low"]) & (preds["target"] <= preds["p_high"])).mean()
    assert 0 <= cov <= 1
