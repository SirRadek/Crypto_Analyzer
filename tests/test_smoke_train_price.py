import numpy as np
import pandas as pd

import ml.time_cv as time_cv
import ml.train_price as tp
import ml.xgb_price as xgb_price
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
        model = xgb_price.xgb.XGBRegressor(
            n_estimators=10,
            max_depth=3,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            tree_method="hist",
            n_jobs=1,
            eval_metric="rmse",
            random_state=42,
        )
        return model

    def small_quant(alpha):
        model = xgb_price.xgb.XGBRegressor(
            n_estimators=10,
            max_depth=3,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            tree_method="hist",
            n_jobs=1,
            objective="reg:quantileerror",
            quantile_alpha=alpha,
            random_state=42,
        )
        return model

    monkeypatch.setattr(xgb_price, "build_reg", small_reg)
    monkeypatch.setattr(xgb_price, "build_quantile", small_quant)

    def small_folds(n_samples, embargo=24):
        return time_cv.time_folds(n_samples, n_splits=2, embargo=embargo)

    monkeypatch.setattr(tp, "time_folds", small_folds)

    _, preds = tp.train_price(df, FEATURE_COLUMNS, outdir=tmp_path)
    assert ((preds["p_hat"] >= preds["p_low"]) & (preds["p_hat"] <= preds["p_high"])).all()
