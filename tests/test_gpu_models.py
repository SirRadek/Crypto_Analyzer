import os
import sys

sys.path.append(os.getcwd())

import logging

import pandas as pd
import pytest
from sklearn.datasets import make_classification, make_regression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBRegressor

from ml.train import _gpu_available as cls_gpu_available
from ml.train import train_model
from ml.train_regressor import (
    _gpu_available as reg_gpu_available,
)
from ml.train_regressor import (
    train_regressor,
)


def _cls_data():
    X, y = make_classification(n_samples=30, n_features=4, random_state=0)
    return pd.DataFrame(X), pd.Series(y)


def _reg_data():
    X, y = make_regression(n_samples=30, n_features=4, random_state=0)
    return pd.DataFrame(X), pd.Series(y)


def test_classifier_gpu_fallback(monkeypatch, caplog):
    X, y = _cls_data()
    caplog.set_level(logging.WARNING)
    monkeypatch.setattr("ml.train._gpu_available", lambda: False)
    model = train_model(X, y, use_gpu=True)
    assert isinstance(model, RandomForestClassifier)
    assert any("falling back to CPU" in r.message for r in caplog.records)


def test_regressor_gpu_fallback(monkeypatch, caplog):
    X, y = _reg_data()
    caplog.set_level(logging.WARNING)
    monkeypatch.setattr("ml.train_regressor._gpu_available", lambda: False)
    model = train_regressor(X, y, use_gpu=True)
    assert isinstance(model, XGBRegressor)
    assert any("falling back to CPU" in r.message for r in caplog.records)


@pytest.mark.skipif(not cls_gpu_available() or not reg_gpu_available(), reason="CUDA not available")
def test_gpu_shapes_equal():
    pytest.importorskip("cuml")
    import cudf  # type: ignore

    Xc, yc = _cls_data()
    Xr, yr = _reg_data()

    cls_gpu = train_model(Xc, yc, use_gpu=True)
    cls_cpu = train_model(Xc, yc)
    preds_gpu_cls = cls_gpu.predict(cudf.from_pandas(Xc.astype("float32")))
    preds_cpu_cls = cls_cpu.predict(Xc)
    if hasattr(preds_gpu_cls, "to_numpy"):
        preds_gpu_cls = preds_gpu_cls.to_numpy()
    assert preds_gpu_cls.shape == preds_cpu_cls.shape

    reg_gpu = train_regressor(Xr, yr, use_gpu=True, params={"n_estimators": 10})
    reg_cpu = train_regressor(Xr, yr, params={"n_estimators": 10})
    preds_gpu_reg = reg_gpu.predict(Xr.astype("float32"))
    preds_cpu_reg = reg_cpu.predict(Xr)
    assert preds_gpu_reg.shape == preds_cpu_reg.shape
