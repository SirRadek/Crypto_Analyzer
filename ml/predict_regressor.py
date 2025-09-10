"""Helpers for making predictions with regression models."""

from .train_regressor import load_regressor
from .ensemble import predict_weighted as predict_weighted_prices


def predict_prices(df, feature_cols, model_path="ml/model_reg.pkl"):
    """Predict prices using a single regressor model."""
    model = load_regressor(model_path)
    X = df[feature_cols]
    return model.predict(X)


__all__ = ["predict_prices", "predict_weighted_prices"]
