import time
from db.db_connector import get_price_data
from analysis.feature_engineering import create_features
from ml.train import train_model, load_model
from ml.predict import predict_ml
from prediction.predictor import combine_predictions
from utils.helpers import ensure_dir_exists
from utils.progress import step, timed, p

SYMBOL = "BTCUSDT"
DB_PATH = "db/data/crypto_data.sqlite"
FEATURE_COLS = ['return_1d', 'sma_7', 'sma_14', 'ema_7', 'ema_14', 'rsi_14']

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
        df = get_price_data(SYMBOL, db_path=DB_PATH)
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
