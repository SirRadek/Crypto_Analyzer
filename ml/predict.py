"""Prediction helpers for classification models."""

from pathlib import Path

from .meta import predict_meta


def predict_ml(
    df,
    feature_cols,
    model_path: str = "ml/meta_model_cls.joblib",
    threshold_path: str = "ml/threshold.json",
):
    """Predict class labels (0/1) using the meta classifier.

    If ``threshold_path`` exists, the stored threshold is used instead of the
    default 0.5.
    """

    return predict_meta(
        df,
        feature_cols,
        model_path,
        threshold_path=threshold_path if Path(threshold_path).exists() else None,
    )


def predict_ml_proba(df, feature_cols, model_path: str = "ml/meta_model_cls.joblib"):
    """Return probability of the positive class (price up)."""
    return predict_meta(df, feature_cols, model_path, proba=True)


__all__ = ["predict_ml", "predict_ml_proba"]
