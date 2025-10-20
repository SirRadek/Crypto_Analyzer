from __future__ import annotations

import numpy as np
import pandas as pd

from crypto_analyzer.models.core import EnsembleModel, RandomForestModel


def test_random_forest_model_train_predict(tmp_path) -> None:
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(100, 4)), columns=list("abcd"))
    y = rng.integers(0, 2, size=100)

    model = RandomForestModel(n_estimators=10, random_state=0)
    model.train(X, y)

    preds = model.predict(X)
    assert isinstance(preds, np.ndarray)
    assert preds.shape == (100,)

    path = tmp_path / "rf.joblib"
    model.save(path)
    loaded = RandomForestModel.load(path)
    loaded_preds = loaded.predict(X)
    np.testing.assert_array_equal(loaded_preds, preds)


def test_ensemble_model_mean_between_base_predictions() -> None:
    rng = np.random.default_rng(1)
    X = pd.DataFrame(rng.normal(size=(50, 3)), columns=list("xyz"))
    y = (X.sum(axis=1) > 0).astype(int).to_numpy()

    base1 = RandomForestModel(n_estimators=5, random_state=1)
    base2 = RandomForestModel(n_estimators=5, random_state=2)
    ensemble = EnsembleModel([base1, base2], strategy="mean")

    ensemble.train(X, y)

    preds_base1 = base1.predict(X)
    preds_base2 = base2.predict(X)
    preds_ensemble = ensemble.predict(X)

    assert preds_ensemble.shape == (50,)
    lower = np.minimum(preds_base1, preds_base2)
    upper = np.maximum(preds_base1, preds_base2)
    assert np.all(preds_ensemble >= lower)
    assert np.all(preds_ensemble <= upper)
