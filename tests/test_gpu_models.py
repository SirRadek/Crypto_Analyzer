import logging
import os
import sys

import pandas as pd
import pytest
import xgboost as xgb
from pandas import DataFrame, Series
from sklearn.datasets import make_regression
from xgboost import XGBRegressor

sys.path.append(os.getcwd())

from ml.train_regressor import train_regressor


def _reg_data() -> tuple[DataFrame, Series]:
    X, y = make_regression(n_samples=30, n_features=4, random_state=0)
    return pd.DataFrame(X).astype("float32"), pd.Series(y).astype("float32")


def test_regressor_gpu_fallback(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    X, y = _reg_data()
    caplog.set_level(logging.WARNING)

    orig_train = xgb.train

    def fake_train(params, *args, **kwargs):
        if params.get("device") == "cuda":
            raise xgb.core.XGBoostError("no cuda")
        return orig_train(params, *args, **kwargs)

    monkeypatch.setattr(xgb, "train", fake_train)
    model = train_regressor(X, y, use_gpu=True, params={"n_estimators": 10})
    assert isinstance(model, XGBRegressor)
    assert any("falling back to CPU" in r.message for r in caplog.records)
