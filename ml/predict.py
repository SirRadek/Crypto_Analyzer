"""Prediction helpers for classification models."""

import json
from .meta import predict_meta


def _load_threshold(path: str) -> float:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return float(json.load(f)["threshold"])
    except Exception:
        return 0.5


def predict_ml(
    df,
    feature_cols,
    model_path="ml/meta_model_cls.joblib",
    threshold_path="ml/threshold.json",
):
    """Predict class labels (0/1) using the calibrated meta classifier."""

    probas = predict_meta(df, feature_cols, model_path, proba=True)
    threshold = _load_threshold(threshold_path)
    return (probas >= threshold).astype(int)


def predict_ml_proba(df, feature_cols, model_path="ml/meta_model_cls.joblib"):
    """Return calibrated probability of the positive class (price up)."""
    return predict_meta(df, feature_cols, model_path, proba=True)


__all__ = ["predict_ml", "predict_ml_proba"]
