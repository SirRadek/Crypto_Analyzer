import os
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

import ml.train_price as tp
from crypto_analyzer.schemas import FeatureConfig, TrainConfig


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

    return small_reg, small_bound


def test_api_contract(tmp_path, monkeypatch):
    small_reg, small_bound = _small_models()
    monkeypatch.setattr(tp, "build_reg", small_reg)
    monkeypatch.setattr(tp, "build_bound", small_bound)

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
    config = TrainConfig(
        horizon_min=120,
        embargo=24,
        target_kind="log",
        xgb_params={"reg": {}, "bound": {}},
        quantiles={"low": 0.1, "high": 0.9},
        fees={"taker": 0.0004},
        features=FeatureConfig(path=Path("analysis/feature_list.json")),
        n_jobs=1,
    )
    tp.train_price(df, config, outdir=model_dir)
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
    assert j["p_low"] <= j["p_hat"] <= j["p_high"]
