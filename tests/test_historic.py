import numpy as np
import pandas as pd

import ml.train_historic as th
import ml.xgb_price as xgb_price


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


def test_train_historic(tmp_path, monkeypatch):
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
    data_path = tmp_path / "data.csv"
    df.to_csv(data_path, index=False)
    outdir = tmp_path / "models"
    th.main(
        [
            "--horizon-min",
            "120",
            "--start-date",
            "2024-01-01",
            "--end-date",
            "2024-01-10",
            "--features",
            "ml/feature_list.json",
            "--outdir",
            str(outdir),
            "--data",
            str(data_path),
        ]
    )
    expected = outdir / "2024-01-01_2024-01-10" / "model_meta.json"
    assert expected.exists()
