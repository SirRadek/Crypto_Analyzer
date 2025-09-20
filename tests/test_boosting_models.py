import sys
import types

import numpy as np
import pytest

from crypto_analyzer.models.boosting import (
    BoostingConfig,
    prepare_sample_weights,
    train_gradient_boosting,
)


def test_prepare_sample_weights_balanced():
    y = np.array([0, 0, 0, 1, 1])
    weights = prepare_sample_weights(y, "balanced")
    assert pytest.approx(weights[y == 0].mean(), rel=1e-6) != pytest.approx(
        weights[y == 1].mean(), rel=1e-6
    )


def test_train_gradient_boosting_uses_class_weights(monkeypatch):
    X = np.random.rand(20, 3)
    y = np.array([0] * 15 + [1] * 5)

    class DummyModel:
        def __init__(self, **params):
            self.params = params
            self._fit_called = False

        def fit(self, X, y, sample_weight=None, eval_set=None, verbose=None):
            self._fit_called = True
            self.sample_weight = sample_weight

        def predict_proba(self, X):
            return np.tile(np.array([[0.6, 0.4]]), (len(X), 1))

    dummy_module = types.SimpleNamespace(XGBClassifier=DummyModel)
    monkeypatch.setitem(sys.modules, "xgboost", dummy_module)

    result = train_gradient_boosting(
        X,
        y,
        config=BoostingConfig(test_size=0.0, class_weight="balanced"),
    )
    model = result.model
    assert isinstance(model, DummyModel)
    assert model._fit_called
    assert "scale_pos_weight" in model.params
    assert result.metrics["accuracy"] >= 0.0

