"""Prediction helpers for classification models."""

from .train import load_model
from .ensemble import predict_weighted


def predict_ml(df, feature_cols, model_path="ml/model.joblib"):
    """Predict class labels (0/1) using a trained classifier."""
    model = load_model(model_path)
    X = df[feature_cols]
    return model.predict(X)


def predict_ml_proba(df, feature_cols, model_path="ml/model.joblib"):
    """Return probability of the positive class (price up)."""
    model = load_model(model_path)
    X = df[feature_cols]
    return model.predict_proba(X)[:, 1]


__all__ = ["predict_ml", "predict_ml_proba", "predict_weighted"]