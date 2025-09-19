import json
from pathlib import Path

import numpy as np
import pandas as pd

import ml.train_price as tp
import ml.xgb_price as xgb_price
from crypto_analyzer.schemas import FeatureConfig, TrainConfig


def test_model_meta(tmp_path, monkeypatch):
    rng = np.random.default_rng(0)
    n = 300
    ts = pd.date_range("2024-01-01", periods=n, freq="15min", tz="UTC")
    close = 100 + rng.normal(scale=1, size=n).cumsum()
    high = close + rng.random(n)
    low = close - rng.random(n)
    open_ = (high + low) / 2
    volume = rng.random(n) + 1
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

    def small_reg():
        params = {
            "max_depth": 3,
            "eta": 0.1,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "tree_method": "hist",
            "eval_metric": "rmse",
            "nthread": 1,
            "seed": 42,
        }
        return params, 10

    monkeypatch.setattr(xgb_price, "build_reg", small_reg)
    monkeypatch.setattr(tp, "build_reg", small_reg)
    config = TrainConfig(
        horizon_min=120,
        embargo=24,
        target_kind="log",
        xgb_params={"reg": {}, "bound": {}},
        bounds={"low": 0.1, "high": 0.9},
        fees={"taker": 0.0004},
        features=FeatureConfig(path=Path("analysis/feature_list.json")),
        n_jobs=1,
    )
    metrics, _ = tp.train_price(df, config, outdir=tmp_path)
    meta_path = tmp_path / "model_meta.json"
    assert meta_path.exists()
    with open(meta_path, encoding="utf-8") as f:
        meta = json.load(f)
    assert meta["data"]["n_samples"] > 0
    assert meta["horizon_min"] == 120
    assert "metrics" in meta
