from .train_regressor import load_regressor

def predict_prices(df, feature_cols, model_path="ml/model_reg.pkl"):
    model = load_regressor(model_path)
    X = df[feature_cols]
    return model.predict(X)
