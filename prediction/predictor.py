from analysis.rules import combined_signal
from ml.predict import predict_ml


def combine_predictions(
    df,
    feature_cols,
    *,
    model_paths=None,
    method="majority",
    use_meta_only=True,
    usage_path="ml/model_usage.json",
):
    """Combines rule-based and ML predictions.

    When ``use_meta_only`` is True (default), only the meta model is used.
    Setting it to False enables the legacy weighted ensemble across
    ``model_paths`` for backward compatibility.

    Parameters
    ----------
    df : pandas.DataFrame
        Data for making predictions.
    feature_cols : list[str]
        Features used by the meta model.
    model_paths : list[str] | None
        Optional paths to base models for legacy ensembling.
    method : str, optional
        - ``"majority"``: 1 if at least one predicts up, else 0
        - ``"strict"``: 1 only if both agree
    use_meta_only : bool, optional
        If True, skip loading any base models and rely solely on the meta
        classifier.  If False, ``model_paths`` must be provided.
    usage_path : str, optional
        Location of JSON file tracking model usage counts for the legacy
        ensemble.

    Returns
    -------
    pandas.Series
        Final combined predictions.
    """
    rule_preds = combined_signal(df)
    if use_meta_only or not model_paths:
        ml_preds = predict_ml(df, feature_cols)
    else:
        from ml.predict import predict_weighted

        ml_preds = predict_weighted(df, feature_cols, model_paths, usage_path=usage_path)

    if method == "majority":
        final_pred = ((rule_preds + ml_preds) > 0).astype(int)
    elif method == "strict":
        final_pred = ((rule_preds + ml_preds) == 2).astype(int)
    else:
        raise ValueError("Unknown combination method")

    return final_pred
