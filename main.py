import time

import pandas as pd

from analysis.feature_engineering import create_features, FEATURE_COLUMNS
from db.db_connector import get_price_data
from ml.predict import predict_ml
from ml.train import train_model, load_model
from prediction.predictor import combine_predictions
from utils.helpers import ensure_dir_exists
from utils.progress import step, timed, p

SYMBOL = "BTCUSDT"
DB_PATH = "db/data/crypto_data.sqlite"
FEATURE_COLS = FEATURE_COLUMNS

def prepare_target(df, forward_steps=1):
    df = df.copy()
    df['target'] = (df['close'].shift(-forward_steps) > df['close']).astype(int)
    df = df.dropna()
    return df

def main(train=True):
    ensure_dir_exists("db/data")
    start = time.perf_counter()

    step(1, 7, "Loading data from DB")
    with timed("Load"):
        start_ts = int((pd.Timestamp.utcnow() - pd.Timedelta(days=5 * 365)).timestamp() * 1000)
        df = get_price_data(SYMBOL, start_ts=start_ts, db_path=DB_PATH)
        p(f"  -> rows={len(df)}, cols={len(df.columns)}")

    step(2, 7, "Feature engineering")
    with timed("Features"):
        df = create_features(df)
        p(f"  -> rows after features={len(df)}")

    step(3, 7, "Preparing classification target (up/down)")
    with timed("Target"):
        df = prepare_target(df)
        p(f"  -> rows after target & dropna={len(df)}")

    X = df[FEATURE_COLS]
    y = df['target']
    p(f"  -> X shape={X.shape}, y shape={y.shape}")

    step(4, 7, "Training / Loading model")
    if train:
        with timed("Train"):
            model = train_model(X, y)
    else:
        with timed("Load model"):
            model = load_model()

    step(5, 7, "Predicting with ML")
    with timed("Predict ML"):
        ml_preds = predict_ml(df, FEATURE_COLS)

    step(6, 7, "Combining rule-based & ML")
    with timed("Combine"):
        combined_preds = combine_predictions(df, FEATURE_COLS)

    step(7, 7, "Printing last 10 predictions")
    tail = df[['close']].copy()
    tail['ml_pred'] = ml_preds
    tail['combined_pred'] = combined_preds
    p(tail.tail(10).to_string(index=True))

    total = time.perf_counter() - start
    p(f"Done in {total:.2f}s")

if __name__ == "__main__":
    main()
