"""Helpers for making predictions with regression models."""

from .meta import predict_meta



def predict_prices(df, feature_cols, model_path="ml/meta_model_reg.joblib"):
    """Predict prices using the meta regressor."""
    return predict_meta(df, feature_cols, model_path)



__all__ = ["predict_prices"]
