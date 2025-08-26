from datetime import datetime
import pandas as pd

from db.db_connector import get_price_data
from analysis.feature_engineering import create_features, FEATURE_COLUMNS
from ml.train_regressor import train_regressor, load_regressor
from ml.predict_regressor import predict_prices
from db.predictions_store import create_predictions_table, save_predictions
from utils.progress import step, timed, p, bar

SYMBOL   = "BTCUSDT"
INTERVAL = "5m"
DB_PATH  = "db/data/crypto_data.sqlite"

FEATURE_COLS = FEATURE_COLUMNS

START = "2024-06-01 00:00:00"
END   = "2024-06-30 23:55:00"
FORWARD_STEPS = 1

def prepare_reg_target(df, forward_steps=1):
    df = df.copy()
    df["target_close"] = df["close"].shift(-forward_steps)
    return df

def main(train=True):
    step(1, 7, f"Loading data for {SYMBOL} from DB")
    with timed("Load"):
        df = get_price_data(SYMBOL, db_path=DB_PATH)
        p(f"  -> rows={len(df)}")

    step(2, 7, "Feature engineering")
    with timed("Features"):
        df = create_features(df)
        p(f"  -> rows after features={len(df)}")

    step(3, 7, f"Preparing regression target (close[t+{FORWARD_STEPS}])")
    with timed("Target"):
        df = prepare_reg_target(df, FORWARD_STEPS)
        p(f"  -> non-null target rows so far={df['target_close'].notna().sum()}")

    start_dt, end_dt = pd.to_datetime(START), pd.to_datetime(END)
    train_df = df[(df["timestamp"] < start_dt) & df["target_close"].notna()]
    pred_df  = df[(df["timestamp"] >= start_dt) & (df["timestamp"] <= end_dt)].copy()
    p(f"  -> train rows={len(train_df)}, predict rows in window={len(pred_df)}")

    if train_df.empty or pred_df.empty:
        p("No data for training or prediction window."); return

    X_train, y_train = train_df[FEATURE_COLS], train_df["target_close"]

    step(4, 7, "Training / Loading regressor")
    if train:
        with timed("Train regressor"):
            model = train_regressor(X_train, y_train)
    else:
        with timed("Load regressor"):
            model = load_regressor()

    step(5, 7, "Predicting prices in the window")
    with timed("Predict"):
        y_pred = predict_prices(pred_df, FEATURE_COLS)
        p(f"  -> predictions={len(y_pred)}")

    step(6, 7, "Preparing rows and saving to table 'prediction'")
    with timed("Save"):
        interval_to_min = {"1m":1,"3m":3,"5m":5,"15m":15,"30m":30,"1h":60,"2h":120,"4h":240,"1d":1440}
        horizon_minutes = interval_to_min[INTERVAL] * FORWARD_STEPS

        pred_df["prediction_time_ms"] = (pred_df["timestamp"].astype("int64") // 10**6)
        target_dt = pred_df["timestamp"] + pd.Timedelta(minutes=horizon_minutes)
        pred_df["target_time_ms"] = (target_dt.astype("int64") // 10**6)

        created_at = datetime.utcnow().isoformat()
        rows = []
        for i, yp in enumerate(bar(y_pred, desc="Building rows", unit="row")):
            rows.append((
                SYMBOL, INTERVAL, FORWARD_STEPS,
                int(pred_df.iloc[i]["prediction_time_ms"]),
                int(pred_df.iloc[i]["target_time_ms"]),
                float(yp),
                None, None,
                "RandomForestRegressor", "v1_features",
                created_at
            ))

        create_predictions_table(DB_PATH, "prediction")
        save_predictions(rows, DB_PATH, "prediction")
        p(f"  -> saved {len(rows)} rows to db/data/crypto_data.sqlite: table 'prediction'")

    step(7, 7, "All done")
    p(f"Window: {START} → {END}, horizon_steps={FORWARD_STEPS}")

if __name__ == "__main__":
    main(train=True)
