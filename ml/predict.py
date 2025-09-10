"""Prediction helpers for classification models."""

from .train import load_model


def predict_ml(df, feature_cols, model_path="ml/meta_model_cls.joblib"):
    """Predict class labels (0/1) using the meta classifier."""
    model = load_model(model_path)
    X = df[feature_cols]
    return model.predict(X)


def predict_ml_proba(df, feature_cols, model_path="ml/meta_model_cls.joblib"):
    """Return probability of the positive class (price up)."""
    model = load_model(model_path)
    X = df[feature_cols]
    return model.predict_proba(X)[:, 1]


__all__ = ["predict_ml", "predict_ml_proba"]