import time
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import pandas as pd

from analysis.feature_engineering import create_features, FEATURE_COLUMNS
from analysis.compare_predictions import backfill_actuals_and_errors
from db.db_connector import get_price_data
from db.predictions_store import create_predictions_table, save_predictions
from deleting_SQL import delete_old_records
from ml.train import train_model, load_model
from ml.train_regressor import train_regressor, load_regressor
from utils.config import CONFIG
from utils.helpers import ensure_dir_exists
from utils.progress import step, timed, p

SYMBOL = CONFIG.symbol
DB_PATH = CONFIG.db_path
FEATURE_COLS = FEATURE_COLUMNS
TABLE_PRED = CONFIG.table_pred
INTERVAL = CONFIG.interval
# predict 2 hours (120 minutes) ahead -> 24 five-minute steps
FORWARD_STEPS = CONFIG.forward_steps
INTERVAL_TO_MIN = {"1m":1,"3m":3,"5m":5,"15m":15,"30m":30,"1h":60,"2h":120,"4h":240,"1d":1440}
PRAGUE_TZ = ZoneInfo("Europe/Prague")
FEATURES_VERSION = "ext_v1"


def _created_at_iso():
    return datetime.now(timezone.utc).isoformat()


def _delete_future_predictions(db_path: str, symbol: str, from_ms: int, table_name: str) -> int:
    import sqlite3

    with sqlite3.connect(db_path) as conn:
        cur = conn.cursor()
        cur.execute(
            f"DELETE FROM {table_name} WHERE symbol = ? AND prediction_time_ms >= ?",
            (symbol, int(from_ms)),
        )
        deleted = cur.rowcount if cur.rowcount is not None else 0
        conn.commit()
        cur.execute("PRAGMA optimize;")
        conn.commit()
    return deleted


def prepare_targets(df, forward_steps=1):
    df = df.copy()
    df["target_cls"] = (df["close"].shift(-forward_steps) > df["close"]).astype(int)
    df["target_reg"] = df["close"].shift(-forward_steps)
    df = df.dropna(subset=["target_cls", "target_reg"])
    return df

def main(train=True):
    ensure_dir_exists("db/data")
    start = time.perf_counter()

    step(1, 8, "Import latest data")
    try:
        from db.btc_import import import_latest_data
        import_latest_data()
    except Exception as exc:
        p(f"btc_import failed: {exc}")

    step(2, 8, "Loading data from DB")
    with timed("Load"):
        start_ts = int((pd.Timestamp.utcnow() - pd.Timedelta(days=5 * 365)).timestamp() * 1000)
        df = get_price_data(SYMBOL, start_ts=start_ts, db_path=DB_PATH)
        p(f"  -> rows={len(df)}, cols={len(df.columns)}")

    step(3, 8, "Feature engineering")
    with timed("Features"):
        df = create_features(df)
        p(f"  -> rows after features={len(df)}")

    full_df = df.copy()

    step(4, 8, "Preparing targets")
    with timed("Target"):
        df = prepare_targets(df, forward_steps=FORWARD_STEPS)
        p(f"  -> rows after targets & dropna={len(df)}")

    X = df[FEATURE_COLS]
    y_cls = df["target_cls"]
    y_reg = df["target_reg"]
    p(f"  -> X shape={X.shape}, y_cls shape={y_cls.shape}, y_reg shape={y_reg.shape}")

    step(5, 8, "Backfill predictions & train/load models")
    create_predictions_table(DB_PATH, TABLE_PRED)
    backfill_actuals_and_errors(db_path=DB_PATH, table_pred=TABLE_PRED, symbol=SYMBOL)
    last_ts = full_df["timestamp"].max()
    _delete_future_predictions(DB_PATH, SYMBOL, int(last_ts.value // 1_000_000), TABLE_PRED)

    if train:
        with timed("Train classifier"):
            cls_model = train_model(X, y_cls)
        with timed("Train regressor"):
            reg_model = train_regressor(X, y_reg)
    else:
        with timed("Load classifier"):
            cls_model = load_model()
        with timed("Load regressor"):
            reg_model = load_regressor()

    step(6, 8, "Predicting last step")
    with timed("Predict"):
        last_row = full_df.iloc[[-1]]
        X_last = last_row[FEATURE_COLS]
        prob_up = float(cls_model.predict_proba(X_last)[:, 1][0])
        reg_pred = float(reg_model.predict(X_last)[0])
        last_close = float(last_row["close"].iloc[0])
        combined_price = last_close + (reg_pred - last_close) * prob_up

    step(7, 8, "Save prediction")
    pred_time = last_row["timestamp"].iloc[0]
    target_time = pred_time + pd.Timedelta(minutes=FORWARD_STEPS * INTERVAL_TO_MIN[INTERVAL])
    pred_local = pd.Timestamp(pred_time, tz="UTC").tz_convert(PRAGUE_TZ)
    targ_local = pd.Timestamp(target_time, tz="UTC").tz_convert(PRAGUE_TZ)
    row = (
        SYMBOL,
        INTERVAL,
        FORWARD_STEPS,
        int(pred_time.value // 1_000_000),
        int(target_time.value // 1_000_000),
        pred_local.strftime("%Y-%m-%d %H:%M:%S"),
        targ_local.strftime("%Y-%m-%d %H:%M:%S"),
        float(combined_price),
        float(reg_pred),
        float(prob_up),
        None,
        None,
        "RF_cls_reg",
        FEATURES_VERSION,
        _created_at_iso(),
    )
    save_predictions([row], DB_PATH, TABLE_PRED)
    p("Saved latest prediction to DB.")

    step(8, 8, "Cleanup old records")
    prices_del, preds_del = delete_old_records(DB_PATH)
    p(f"  -> deleted {prices_del} prices rows and {preds_del} prediction rows")

    total = time.perf_counter() - start
    p(f"Done in {total:.2f}s")

if __name__ == "__main__":
    main()