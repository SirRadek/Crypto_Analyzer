"""Prediction helpers for classification models."""

from .meta import predict_meta



def predict_ml(df, feature_cols, model_path="ml/meta_model_cls.joblib"):
    """Predict class labels (0/1) using the meta classifier."""
    return predict_meta(df, feature_cols, model_path)



def predict_ml_proba(df, feature_cols, model_path="ml/meta_model_cls.joblib"):
    """Return probability of the positive class (price up)."""
    return predict_meta(df, feature_cols, model_path, proba=True)


__all__ = ["predict_ml", "predict_ml_proba"]
