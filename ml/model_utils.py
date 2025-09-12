from __future__ import annotations

import os
from typing import Any, List

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score


# ---------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------

def save_model(model: Any, path: str) -> None:
    """Persist model to path using joblib."""
    joblib.dump(model, path)
    print(f"Model saved to {path}")


def load_model(path: str) -> Any:
    """Load a model from disk."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found at {path}")
    return joblib.load(path)


# ---------------------------------------------------------------------
# Feature alignment
# ---------------------------------------------------------------------

def _expected_feature_names(model: Any) -> List[str]:
    """
    Derive the feature-name contract of a fitted model.

    Priority:
      1) XGBoost booster.feature_names
      2) sklearn-like feature_names_in_
      3) n_features_in_ → f0..f{n-1}
    """
    # XGBoost
    try:
        booster = getattr(model, "get_booster", None)
        if booster is not None:
            bst = booster()
            names = getattr(bst, "feature_names", None)
            if names:
                return list(names)
    except Exception:
        pass

    # sklearn
    names = getattr(model, "feature_names_in_", None)
    if names is not None:
        return [str(n) for n in list(names)]

    n = int(getattr(model, "n_features_in_", 0) or 0)
    if n > 0:
        return [f"f{i}" for i in range(n)]

    raise RuntimeError("Cannot determine expected feature names for the provided model.")


def match_model_features(
    df: pd.DataFrame,
    model: Any,
    *,
    fill_value: float = 0.0,
    dtype: str = "float32",
) -> pd.DataFrame:
    """
    Align df columns to the model’s expected features:
      - add missing columns with fill_value
      - drop unknown columns
      - reorder to the training order
      - cast to float32 by default
    """
    expected = _expected_feature_names(model)
    X = df.copy()

    # add missing
    missing = [c for c in expected if c not in X.columns]
    if missing:
        for c in missing:
            X[c] = fill_value

    # drop extras
    extra = [c for c in X.columns if c not in expected]
    if extra:
        X = X.drop(columns=extra)

    # reorder and cast
    X = X[expected].astype(dtype, copy=False)
    return X


# ---------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------

def evaluate_model(
    model: Any,
    X_test: np.ndarray | pd.DataFrame,
    y_test: np.ndarray | pd.Series,
) -> tuple[float, float]:
    """
    Print basic classification metrics and return (accuracy, f1_weighted).
    For regression tasks, use a dedicated evaluator.
    """
    preds = model.predict(X_test)
    preds_arr = np.asarray(preds)
    y_arr = np.asarray(y_test)
    acc = float(accuracy_score(y_arr, preds_arr))
    f1 = float(f1_score(y_arr, preds_arr, average="weighted"))
    print("Accuracy:", acc)
    print("F1 score:", f1)
    print("Classification report:\n", classification_report(y_arr, preds_arr))
    print("Confusion matrix:\n", confusion_matrix(y_arr, preds_arr))
    return acc, f1
