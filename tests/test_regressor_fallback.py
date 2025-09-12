import numpy as np

from ml import train_regressor as tr


class DummyRF:
    def __init__(self, **params):
        self.n_estimators = params.get("n_estimators")

    def fit(self, X, y, sample_weight=None):
        if self.n_estimators == 5:
            raise MemoryError("boom")
        self.fitted_ = True


def test_memory_fallback(monkeypatch, tmp_path):
    monkeypatch.setattr(tr, "RandomForestRegressor", DummyRF)
    X = np.array([[0.0], [1.0]])
    y = np.array([0.0, 1.0])
    model_path = tmp_path / "m.joblib"
    model = tr.train_regressor(X, y, model_path=str(model_path), fallback_estimators=(5, 3))
    assert isinstance(model, DummyRF)
    assert model.n_estimators == 3
    assert model_path.exists()
