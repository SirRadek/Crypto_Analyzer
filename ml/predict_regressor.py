"""Helpers for making predictions with regression models."""

from .train_regressor import load_regressor


def predict_prices(df, feature_cols, model_path="ml/meta_model_reg.joblib"):
    """Predict prices using the meta regressor."""
    model = load_regressor(model_path)
    X = df[feature_cols]
    return model.predict(X)


__all__ = ["predict_prices"]
