import numpy as np

from ml import train_regressor as tr


class DummyBooster:
    attempts: list[dict[str, object]] = []

    def predict(self, data):  # pragma: no cover - not used
        return np.zeros(getattr(data, "num_row", lambda: len(data))())


def dummy_train(params, dtrain, num_boost_round, evals, early_stopping_rounds):
    assert isinstance(dtrain, tr.xgb.DMatrix)
    assert dtrain.get_label().dtype == np.float32
    DummyBooster.attempts.append(
        {
            "tree_method": params.get("tree_method"),
            "nthread": params.get("nthread"),
            "n_estimators": num_boost_round,
            "max_depth": params.get("max_depth"),
        }
    )
    if len(DummyBooster.attempts) < 4:
        raise MemoryError("boom")
    return DummyBooster()


def test_memory_fallback(monkeypatch, tmp_path):
    monkeypatch.setattr(tr.xgb, "train", dummy_train)
    monkeypatch.setattr(tr, "_gpu_available", lambda: True)
    X = np.arange(8, dtype=np.float32).reshape(-1, 1)
    y = np.arange(8, dtype=np.float32)
    model_path = tmp_path / "m.joblib"
    model = tr.train_regressor(X, y, model_path=str(model_path), use_gpu=True)
    assert isinstance(model, tr.xgb.XGBRegressor)
    assert DummyBooster.attempts[0]["tree_method"] == "gpu_hist"
    assert DummyBooster.attempts[1]["tree_method"] == "hist"
    assert DummyBooster.attempts[1]["nthread"] == 1
    assert DummyBooster.attempts[2]["n_estimators"] < DummyBooster.attempts[1]["n_estimators"]
    assert DummyBooster.attempts[3]["max_depth"] < DummyBooster.attempts[2]["max_depth"]
    assert model_path.exists()
