import numpy as np

from ml import train_regressor as tr


class DummyXGB:
    attempts: list[dict[str, object]] = []

    def __init__(self, **params):
        self.params = params

    def fit(self, X, y, sample_weight=None):
        DummyXGB.attempts.append(self.params.copy())
        assert X.dtype == np.float32
        assert y.dtype == np.float32
        if len(DummyXGB.attempts) < 3:
            raise MemoryError("boom")
        self.fitted_ = True


def test_memory_fallback(monkeypatch, tmp_path):
    monkeypatch.setattr(tr, "XGBRegressor", DummyXGB)
    monkeypatch.setattr(tr, "_gpu_available", lambda: True)
    X = np.array([[0.0], [1.0]], dtype=np.float32)
    y = np.array([0.0, 1.0], dtype=np.float32)
    model_path = tmp_path / "m.joblib"
    model = tr.train_regressor(X, y, model_path=str(model_path), use_gpu=True)
    assert isinstance(model, DummyXGB)
    assert DummyXGB.attempts[0]["n_jobs"] == -1
    assert DummyXGB.attempts[0]["tree_method"] == "gpu_hist"
    assert DummyXGB.attempts[1]["n_jobs"] == 1
    assert DummyXGB.attempts[1]["tree_method"] == "gpu_hist"
    assert DummyXGB.attempts[2]["n_jobs"] == 1
    assert DummyXGB.attempts[2]["tree_method"] == "hist"
    assert model_path.exists()
