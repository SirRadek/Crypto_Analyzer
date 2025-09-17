import numpy as np
import pandas as pd
import yaml

import ml.train_historic as th
import ml.train_price as tp
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

    return small_reg


def test_train_historic(tmp_path, monkeypatch):
    small_reg = _small_models()
    monkeypatch.setattr(xgb_price, "build_reg", small_reg)
    monkeypatch.setattr(tp, "build_reg", small_reg)

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
    cfg = {
        "horizon_min": 120,
        "embargo": 24,
        "target_kind": "log",
        "xgb_params": {"reg": {}, "bound": {}},
        "bounds": {"low": 0.1, "high": 0.9},
        "fees": {"taker": 0.0004},
        "features": {"path": "analysis/feature_list.json"},
        "n_jobs": 1,
    }
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))
    th.main(
        [
            "--config",
            str(cfg_path),
            "--start-date",
            "2024-01-01",
            "--end-date",
            "2024-01-10",
            "--outdir",
            str(outdir),
            "--data",
            str(data_path),
        ]
    )
    expected = outdir / "2024-01-01_2024-01-10" / "model_meta.json"
    assert expected.exists()
