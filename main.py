import json
import time
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from analysis.feature_engineering import create_features, FEATURE_COLUMNS
from analysis.compare_predictions import backfill_actuals_and_errors
from db.db_connector import get_price_data
from db.predictions_store import create_predictions_table, save_predictions
from deleting_SQL import delete_old_records
from ml.train import train_model, load_model
from ml.train_regressor import train_regressor, load_regressor
from ml.model_utils import match_model_features
from utils.config import CONFIG
from utils.helpers import ensure_dir_exists
from utils.progress import step, timed, p

SYMBOL = CONFIG.symbol
DB_PATH = CONFIG.db_path
FEATURE_COLS = FEATURE_COLUMNS
TABLE_PRED = CONFIG.table_pred
INTERVAL = CONFIG.interval
# number of future five-minute steps to predict, e.g. 24 -> next 2 hours
FORWARD_STEPS = CONFIG.forward_steps
INTERVAL_TO_MIN = {
    "1m": 1,
    "3m": 3,
    "5m": 5,
    "15m": 15,
    "30m": 30,
    "1h": 60,
    "2h": 120,
    "4h": 240,
    "1d": 1440,
}
PRAGUE_TZ = ZoneInfo("Europe/Prague")
FEATURES_VERSION = "ext_v1"

CLS_MODEL_COUNT = 1
REG_MODEL_COUNT = 1
CLS_ACC_PATH = Path("ml/backtest_acc_cls.json")
REG_ACC_PATH = Path("ml/backtest_acc_reg.json")


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



def list_model_paths(pattern: str, count: int):
    paths = []
    indices = []
    for i in range(1, count + 1):
        path = pattern.format(i=i)
        if Path(path).exists():
            paths.append(path)
            indices.append(i)
        else:
            p(f"Missing model {path}")
    return paths, indices


def load_accuracy_weights(path: Path, indices):
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        data = {}
    return [float(data.get(str(i), 1.0)) for i in indices]


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

    step(4, 8, "Backfill predictions & train/load models")
    create_predictions_table(DB_PATH, TABLE_PRED)
    backfill_actuals_and_errors(db_path=DB_PATH, table_pred=TABLE_PRED, symbol=SYMBOL)
    last_ts = full_df["timestamp"].max()
    _delete_future_predictions(DB_PATH, SYMBOL, int(last_ts.value // 1_000_000), TABLE_PRED)
    # Load base models and their validation weights
    cls_paths, cls_indices = list_model_paths("ml/meta_model_cls.joblib", CLS_MODEL_COUNT)
    reg_paths, reg_indices = list_model_paths("ml/meta_model_reg.joblib", REG_MODEL_COUNT)
    cls_weights = load_accuracy_weights(CLS_ACC_PATH, cls_indices)
    reg_weights = load_accuracy_weights(REG_ACC_PATH, reg_indices)

    # Prepare training data including predictions from base models
    horizon_dfs = []
    for horizon in range(1, FORWARD_STEPS + 1):
        df_h = prepare_targets(full_df, forward_steps=horizon)
        df_h["horizon"] = horizon
        horizon_dfs.append(df_h)
    train_df = pd.concat(horizon_dfs, ignore_index=True)

    base_feats = train_df[FEATURE_COLS]
    for path, idx, w in zip(cls_paths, cls_indices, cls_weights):
        model = load_model(model_path=path)
        feats = match_model_features(base_feats, model)
        preds = model.predict_proba(feats)[:, 1] * w
        train_df[f"cls_pred_{idx}"] = preds
        del model
    for path, idx, w in zip(reg_paths, reg_indices, reg_weights):
        model = load_regressor(model_path=path)
        feats = match_model_features(base_feats, model)
        preds = model.predict(feats) * w
        train_df[f"reg_pred_{idx}"] = preds
        del model

    cls_pred_cols = [f"cls_pred_{i}" for i in cls_indices]
    reg_pred_cols = [f"reg_pred_{i}" for i in reg_indices]
    feature_cols_meta = FEATURE_COLS + ["horizon"] + cls_pred_cols + reg_pred_cols
    X_all = train_df[feature_cols_meta]
    y_cls_all = train_df["target_cls"]
    y_reg_all = train_df["target_reg"]

    model_path_cls = "ml/meta_model_cls.joblib"
    model_path_reg = "ml/meta_model_reg.joblib"
    if train:
        with timed("Train meta-classifier"):
            cls_model = train_model(X_all, y_cls_all, model_path=model_path_cls)
        with timed("Train meta-regressor"):
            reg_model = train_regressor(X_all, y_reg_all, model_path=model_path_reg)
    else:
        with timed("Load meta-classifier"):
            cls_model = load_model(model_path=model_path_cls)
        with timed("Load meta-regressor"):
            reg_model = load_regressor(model_path=model_path_reg)

    rows_to_save = []
    last_row = full_df.iloc[[-1]]
    pred_time = last_row["timestamp"].iloc[0]
    pred_local = pd.Timestamp(pred_time, tz="UTC").tz_convert(PRAGUE_TZ)

    # Base model predictions for the latest row
    base_last = last_row[FEATURE_COLS]
    last_base_preds = {}
    for path, idx, w in zip(cls_paths, cls_indices, cls_weights):
        model = load_model(model_path=path)
        feats = match_model_features(base_last, model)
        last_base_preds[f"cls_pred_{idx}"] = float(
            model.predict_proba(feats)[:, 1][0] * w
        )
        del model
    for path, idx, w in zip(reg_paths, reg_indices, reg_weights):
        model = load_regressor(model_path=path)
        feats = match_model_features(base_last, model)
        last_base_preds[f"reg_pred_{idx}"] = float(model.predict(feats)[0] * w)
        del model

    step(5, 8, "Predict horizons")
    for horizon in range(1, FORWARD_STEPS + 1):
        with timed("Predict"):
            X_last = base_last.copy()
            X_last["horizon"] = horizon
            for name, val in last_base_preds.items():
                X_last[name] = val
            prob_up = float(cls_model.predict_proba(X_last[feature_cols_meta])[:, 1][0])
            reg_pred = float(reg_model.predict(X_last[feature_cols_meta])[0])
            last_close = float(last_row["close"].iloc[0])
            combined_price = last_close + (reg_pred - last_close) * prob_up

        target_time = pred_time + pd.Timedelta(minutes=horizon * INTERVAL_TO_MIN[INTERVAL])
        targ_local = pd.Timestamp(target_time, tz="UTC").tz_convert(PRAGUE_TZ)

        row = (
            SYMBOL,
            INTERVAL,
            horizon,
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
        rows_to_save.append(row)

    step(6, 8, "Save predictions")
    save_predictions(rows_to_save, DB_PATH, TABLE_PRED)
    p(f"Saved {len(rows_to_save)} predictions to DB.")

    step(7, 8, "Cleanup old records")
    prices_del, preds_del = delete_old_records(DB_PATH)
    p(f"  -> deleted {prices_del} prices rows and {preds_del} prediction rows")

    total = time.perf_counter() - start
    p(f"Done in {total:.2f}s")

if __name__ == '__main__':
    main()