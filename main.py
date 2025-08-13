import os
from db.db_connector import get_price_data
from analysis.feature_engineering import create_features
from ml.train import train_model, load_model
from ml.predict import predict_ml
from prediction.predictor import combine_predictions
from utils.helpers import ensure_dir_exists

SYMBOL = "BTCUSDT"
DB_PATH = "db/data/crypto_data.sqlite"
FEATURE_COLS = ['return_1d', 'sma_7', 'sma_14', 'ema_7', 'ema_14', 'rsi_14']

def prepare_target(df, forward_steps=1):
    """
    Prepares target variable for ML:
    1 = price will go up in next step(s), 0 = down or unchanged.
    """
    df = df.copy()
    df['target'] = (df['close'].shift(-forward_steps) > df['close']).astype(int)
    df = df.dropna()
    return df

def main(train=True):
    ensure_dir_exists("data")
    # 1. Load data
    df = get_price_data(SYMBOL, db_path=DB_PATH)
    if df.empty:
        print("No data found!")
        return

    # 2. Feature engineering
    df = create_features(df)

    # 3. Prepare target variable for ML (supervised learning)
    df = prepare_target(df)
    X = df[FEATURE_COLS]
    y = df['target']

    # 4. Train or load model
    if train:
        model = train_model(X, y)
    else:
        model = load_model()

    # 5. ML predictions
    ml_preds = predict_ml(df, FEATURE_COLS)

    # 6. Rule-based + combined predictions
    combined_preds = combine_predictions(df, FEATURE_COLS)

    # 7. Example output
    df['ml_pred'] = ml_preds
    df['combined_pred'] = combined_preds
    print(df[['close', 'ml_pred', 'combined_pred']].tail(10))

if __name__ == "__main__":
    main()
