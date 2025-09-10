from analysis.rules import combined_signal
from ml.predict import predict_ml


def combine_predictions(
    df,
    feature_cols,
    method="majority",
):
    """
    Combines rule-based and meta-model predictions.

    Parameters
    ----------
    df : pandas.DataFrame
        Data for making predictions.
    feature_cols : list[str]
        Features used by the meta model.
    method : str, optional
        - ``"majority"``: 1 if at least one predicts up, else 0
        - ``"strict"``: 1 only if both agree

    Returns
    -------
    pandas.Series
        Final combined predictions.
    """
    rule_preds = combined_signal(df)
    ml_preds = predict_ml(df, feature_cols)

    if method == "majority":
        final_pred = ((rule_preds + ml_preds) > 0).astype(int)
    elif method == "strict":
        final_pred = ((rule_preds + ml_preds) == 2).astype(int)
    else:
        raise ValueError("Unknown combination method")

    return final_pred