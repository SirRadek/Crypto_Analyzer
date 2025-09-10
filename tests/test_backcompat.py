import os
import sys

sys.path.append(os.getcwd())  # noqa: E402

from pathlib import Path  # noqa: E402

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.ensemble import RandomForestClassifier  # noqa: E402

from analysis.rules import combined_signal  # noqa: E402
from ml.predict import predict_weighted  # noqa: E402
from prediction.predictor import combine_predictions  # noqa: E402


def test_legacy_ensemble_path(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        {
            "rsi_14": rng.uniform(0, 100, 20),
            "sma_7": rng.normal(size=20),
            "sma_14": rng.normal(size=20),
        }
    )
    feature_cols = ["rsi_14", "sma_7", "sma_14"]
    X = df[feature_cols]
    y = rng.integers(0, 2, size=len(df))
    model = RandomForestClassifier(n_estimators=5, random_state=0)
    model.fit(X, y)
    model_path = tmp_path / "base.joblib"
    joblib.dump(model, model_path)

    usage_path = tmp_path / "usage.json"
    ml_preds = predict_weighted(
        df, feature_cols, [str(model_path)], usage_path=usage_path
    )
    rule_preds = combined_signal(df)
    expected = ((rule_preds + ml_preds) > 0).astype(int)

    res = combine_predictions(
        df,
        feature_cols,
        model_paths=[str(model_path)],
        use_meta_only=False,
        usage_path=usage_path,
    )
    assert res.equals(expected)
