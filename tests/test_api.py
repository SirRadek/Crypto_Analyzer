import os

import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

import ml.train_price as tp
import ml.xgb_price as xgb_price
from analysis.feature_engineering import FEATURE_COLUMNS


def _small_models():
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

    def small_quant(alpha):
        params = {
            "max_depth": 2,
            "eta": 0.1,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "tree_method": "hist",
            "objective": "reg:quantileerror",
            "quantile_alpha": alpha,
            "nthread": 1,
            "seed": 42,
        }
        return params, 5

    return small_reg, small_quant


def test_api(tmp_path, monkeypatch):
    small_reg, small_quant = _small_models()
    monkeypatch.setattr(xgb_price, "build_reg", small_reg)
    monkeypatch.setattr(xgb_price, "build_quantile", small_quant)

    rng = np.random.default_rng(0)
    n = 300
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

    model_dir = tmp_path / "models"
    tp.train_price(df, FEATURE_COLUMNS, outdir=model_dir)
    data_path = tmp_path / "data.csv"
    df.to_csv(data_path, index=False)

    os.environ["MODEL_DIR"] = str(model_dir)
    os.environ["DATA_PATH"] = str(data_path)

    from api.server import app

    client = TestClient(app)
    res = client.get("/predict")
    assert res.status_code == 200
    j = res.json()
    assert {"timestamp", "p_low", "p_hat", "p_high"} <= j.keys()
