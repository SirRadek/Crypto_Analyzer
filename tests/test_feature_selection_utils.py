import json

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from crypto_analyzer.features.selection import (
    SelectionConfig,
    drop_correlated_features,
    export_top_features,
    permutation_importance_by_regime,
)


def test_permutation_importance_by_regime_returns_frame(tmp_path):
    rng = np.random.default_rng(42)
    X = pd.DataFrame(rng.normal(size=(120, 3)), columns=["f1", "f2", "f3"])
    y = pd.Series((rng.random(120) > 0.5).astype(int))
    regimes = pd.Series(np.where(np.arange(120) < 60, "bull", "bear"))
    model = RandomForestClassifier(n_estimators=10, random_state=0)
    model.fit(X, y)

    config = SelectionConfig(n_repeats=2, random_state=0)
    importance = permutation_importance_by_regime(model, X, y, regimes, config=config)
    assert set(importance.columns) == {"feature", "importance", "std", "regime"}
    assert {"bull", "bear"}.issubset(set(importance["regime"]))


def test_drop_correlated_features_and_export(tmp_path):
    data = pd.DataFrame(
        {
            "a": [1, 2, 3, 4],
            "b": [2, 4, 6, 8],
            "c": [1, 2, 1, 3],
        }
    )
    kept = drop_correlated_features(data, threshold=0.9)
    assert "b" not in kept

    scores = {"a": 0.5, "c": 0.2}
    output = tmp_path / "top_features.json"
    export_top_features(scores, horizon="2h", top_n=1, output_path=output)
    payload = json.loads(output.read_text())
    assert payload["horizon"] == "2h"
    assert payload["features"] == ["a"]

