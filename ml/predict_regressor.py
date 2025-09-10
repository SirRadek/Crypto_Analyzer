"""Helpers for making predictions with regression models."""

from .meta import predict_meta


def predict_prices(df, feature_cols, model_path="ml/meta_model_reg.joblib"):
    """Predict prices using the meta regressor."""
    return predict_meta(df, feature_cols, model_path)


def predict_weighted_prices(
    df,
    feature_cols,
    model_paths,
    usage_path="ml/model_usage.json",
):
    """Backward-compatible wrapper for weighted base-regressor ensembling."""

    from .ensemble import predict_weighted as _predict_weighted

    return _predict_weighted(
        df,
        feature_cols,
        model_paths,
        usage_counts_path=usage_path,
    )


__all__ = ["predict_prices", "predict_weighted_prices"]
