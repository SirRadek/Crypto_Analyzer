from .train import load_model

def predict_ml(df, feature_cols, model_path="ml/model.pkl"):
    """
    Loads a model and predicts target labels for given DataFrame.
    - df: DataFrame with features
    - feature_cols: list of column names to use as features
    Returns a numpy array or pandas Series with predictions (e.g., 1=up, 0=down).
    """
    model = load_model(model_path)
    X = df[feature_cols]
    preds = model.predict(X)
    return preds
