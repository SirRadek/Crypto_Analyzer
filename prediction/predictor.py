from analysis.rules import combined_signal
from ml.predict import predict_ml, predict_weighted


def combine_predictions(
    df,
    feature_cols,
    model_paths=None,
    method="majority",
    usage_path="ml/model_usage.json",
):
    """
    Combines rule-based and ML predictions.

    Parameters
    ----------
    df : pandas.DataFrame
        Data for making predictions.
    feature_cols : list[str]
        Features used by the ML models.
    model_paths : list[str] | None
        Paths to multiple model files. If ``None``, a single model at
        ``ml/model.joblib`` is used.
    method : str, optional
        - ``"majority"``: 1 if at least one predicts up, else 0
        - ``"strict"``: 1 only if both agree
    usage_path : str, optional
        Location of JSON file tracking model usage counts.

    Returns
    -------
    pandas.Series
        Final combined predictions.
    """
    rule_preds = combined_signal(df)
    if model_paths is None:
        ml_preds = predict_ml(df, feature_cols)
    else:
        ml_preds = predict_weighted(
            df, feature_cols, model_paths, usage_path=usage_path
        )

    if method == "majority":
        final_pred = ((rule_preds + ml_preds) > 0).astype(int)
    elif method == "strict":
        final_pred = ((rule_preds + ml_preds) == 2).astype(int)
    else:
        raise ValueError("Unknown combination method")

    return final_pred