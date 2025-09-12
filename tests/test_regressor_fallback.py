import os
import sys

import numpy as np

sys.path.append(os.getcwd())

import xgboost as xgb

from ml import train_regressor as tr


class DummyBooster:
    attempts: list[dict[str, int | str | None]] = []

    def predict(self, data: xgb.DMatrix) -> np.ndarray:  # pragma: no cover - not used
        return np.zeros(getattr(data, "num_row", lambda: len(data))())


def dummy_train(
    params: dict[str, object],
    dtrain: xgb.DMatrix,
    num_boost_round: int,
    evals: list[tuple[xgb.DMatrix, str]],
    early_stopping_rounds: int,
) -> DummyBooster:
    assert isinstance(dtrain, xgb.DMatrix)
    assert dtrain.get_label().dtype == np.float32
    DummyBooster.attempts.append(
        {
            "device": params.get("device"),
            "nthread": params.get("nthread"),
            "n_estimators": num_boost_round,
            "max_depth": params.get("max_depth"),
        }
    )
    if len(DummyBooster.attempts) < 4:
        raise MemoryError("boom")
    return DummyBooster()


def test_memory_fallback(monkeypatch, tmp_path):
    monkeypatch.setattr(xgb, "train", dummy_train)
    X = np.arange(8, dtype=np.float32).reshape(-1, 1)
    y = np.arange(8, dtype=np.float32)
    model_path = tmp_path / "m.joblib"
    model = tr.train_regressor(X, y, model_path=str(model_path), use_gpu=True)
    assert isinstance(model, xgb.XGBRegressor)
    assert DummyBooster.attempts[0]["device"] == "cuda"
    assert DummyBooster.attempts[1]["device"] == "cpu"
    assert DummyBooster.attempts[1]["nthread"] == 1
    assert int(DummyBooster.attempts[2]["n_estimators"]) < int(
        DummyBooster.attempts[1]["n_estimators"]
    )
    assert int(DummyBooster.attempts[3]["max_depth"]) < int(DummyBooster.attempts[2]["max_depth"])
    assert model_path.exists()
