from pathlib import Path

import numpy as np
import pandas as pd

import ml.train_price as tp
import ml.xgb_price as xgb_price
from crypto_analyzer.schemas import FeatureConfig, TrainConfig


def test_xgb_train_smoke(monkeypatch, tmp_path):
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

    def small_quant(alpha: float):
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

    monkeypatch.setattr(xgb_price, "build_reg", small_reg)
    monkeypatch.setattr(xgb_price, "build_quantile", small_quant)

    config = TrainConfig(
        horizon_min=120,
        embargo=24,
        target_kind="log",
        xgb_params={"reg": {}, "quantile": {}},
        quantiles={"low": 0.1, "high": 0.9},
        fees={"taker": 0.0004},
        features=FeatureConfig(path=Path("analysis/feature_list.json")),
        n_jobs=1,
    )

    metrics, preds = tp.train_price(df, config, outdir=tmp_path)
    assert ((preds["p_hat"] >= preds["p_low"]) & (preds["p_hat"] <= preds["p_high"])).all()
    assert "rmse" in metrics
