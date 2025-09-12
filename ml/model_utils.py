from __future__ import annotations

import os
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)


def save_model(model: Any, path: str) -> None:
    """Persist ``model`` to ``path`` using joblib."""
    joblib.dump(model, path)
    print(f"Model saved to {path}")


def load_model(path: str) -> Any:
    """Load a model from disk."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found at {path}")
    return joblib.load(path)


def match_model_features(df: pd.DataFrame, model: Any) -> pd.DataFrame:
    """Align ``df`` columns with features expected by ``model``."""
    feature_names = getattr(model, "feature_names_in_", None)
    if feature_names is None:
        return df
    return df.reindex(columns=feature_names, fill_value=0)


def evaluate_model(
    model: Any, X_test: np.ndarray | pd.DataFrame, y_test: np.ndarray | pd.Series
) -> tuple[float, float]:
    """Print evaluation metrics for classification models."""
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
