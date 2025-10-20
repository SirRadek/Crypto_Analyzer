import numpy as np
import pytest

from crypto_analyzer.eval.engine import BacktestEngine


class MajorityClassifier:
    """Simple classifier that predicts the majority class observed during fit."""

    def __init__(self) -> None:
        self._fitted = False

    def fit(self, X, y):
        y_arr = np.asarray(y)
        self._positive_prob = float(y_arr.mean())
        self._prediction = 1 if self._positive_prob > 0.5 else 0
        self._fitted = True
        return self

    def predict(self, X):
        if not self._fitted:
            raise RuntimeError("Model must be fitted before predicting")
        return np.full(len(X), self._prediction, dtype=int)

    def predict_proba(self, X):
        if not self._fitted:
            raise RuntimeError("Model must be fitted before predicting")
        probs = np.array([1.0 - self._positive_prob, self._positive_prob], dtype=float)
        return np.tile(probs, (len(X), 1))

    def get_params(self, deep: bool = True):  # pragma: no cover - sklearn clone support
        return {}

    def set_params(self, **params):  # pragma: no cover - sklearn clone support
        return self


def test_holdout_engine_returns_expected_metrics():
    X = np.arange(20, dtype=np.float32).reshape(-1, 1)
    y = np.array([0, 1] * 10, dtype=int)
    engine = BacktestEngine(
        mode="holdout",
        metrics=("accuracy", "precision", "recall"),
        validation_fraction=0.5,
    )

    model = MajorityClassifier()
    metrics = engine.evaluate(model, X, y)

    assert set(metrics) == {"accuracy", "precision", "recall"}
    assert metrics["accuracy"] == pytest.approx(0.5)
    assert metrics["precision"] == pytest.approx(0.0)
    assert metrics["recall"] == pytest.approx(0.0)
    assert not hasattr(model, "_positive_prob")


def test_walkforward_engine_aggregates_fold_metrics():
    X = np.arange(24, dtype=np.float32).reshape(-1, 1)
    y = np.array([0, 1, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 0])

    engine = BacktestEngine(
        mode="walkforward",
        metrics=("accuracy", "precision"),
        walkforward_window=6,
    )

    result = engine.run(MajorityClassifier(), X, y)

    assert set(result["metrics"]) == {"accuracy", "precision"}
    assert len(result["details"]) >= 2
    precisions = [fold["precision"] for fold in result["details"]]
    assert result["metrics"]["precision"] == pytest.approx(np.mean(precisions))

