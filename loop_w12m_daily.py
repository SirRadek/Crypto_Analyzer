import sqlite3
from collections import deque
from datetime import datetime, timezone, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from glob import glob

from analysis.compare_predictions import backfill_actuals_and_errors
from analysis.feature_engineering import create_features, FEATURE_COLUMNS
from db.db_connector import get_price_data
from db.predictions_store import create_predictions_table, save_predictions
from ml.train_regressor import train_regressor
from ml.predict_regressor import predict_weighted_prices

try:
    from utils.progress import step, timed, p
except Exception:
    def p(msg): print(msg, flush=True)
    def step(i, n, msg): p(f"[{i}/{n}] {msg}")
    from contextlib import contextmanager
    @contextmanager
    def timed(label): p(f"{label} ..."); yield; p(f"{label} done")

SYMBOL = "BTCUSDT"
INTERVAL = "5m"
DB_PATH = "db/data/crypto_data.sqlite"
TABLE_PRED = "predictions"
TRAIN_WINDOW_Y = 5
FORWARD_STEPS = 1
FEATURE_COLS = FEATURE_COLUMNS
FORWARD_FEATURE_COLS = ["return_1d","sma_7","sma_14","ema_7","ema_14","rsi_14"]
PRAGUE_TZ = ZoneInfo("Europe/Prague")
INTERVAL_TO_MIN = {"1m":1,"3m":3,"5m":5,"15m":15,"30m":30,"1h":60,"2h":120,"4h":240,"1d":1440}
FEATURES_VERSION_FWD = "wf5yD1_future"


def _created_at_iso():
    return datetime.now(timezone.utc).isoformat()


def _ensure_indexes(table_name: str):
    conn = sqlite3.connect(DB_PATH); cur = conn.cursor()
    cur.execute("CREATE INDEX IF NOT EXISTS idx_prices_open_time ON prices(open_time)")
    cur.execute(f"CREATE INDEX IF NOT EXISTS idx_{table_name}_pred_time ON {table_name}(prediction_time_ms)")
    cur.execute(f"CREATE INDEX IF NOT EXISTS idx_{table_name}_target_time ON {table_name}(target_time_ms)")
    cur.execute(f"CREATE INDEX IF NOT EXISTS idx_{table_name}_symbol ON {table_name}(symbol)")
    conn.commit(); conn.close()


def _prepare_reg_target(df: pd.DataFrame, forward_steps: int) -> pd.DataFrame:
    df = df.copy()
    df["target_close"] = df["close"].shift(-forward_steps)
    return df


def _weights_from_errors_for_range(train_df: pd.DataFrame, db_path: str, symbol: str, table_name: str,
                                   alpha: float = 1.0, max_w: float = 3.0) -> pd.Series:
    if train_df.empty:
        return pd.Series(1.0, index=train_df.index)
    t0 = int(train_df["timestamp"].min().value // 1_000_000)
    t1 = int(train_df["timestamp"].max().value // 1_000_000)
    conn = sqlite3.connect(db_path)
    q = f"""
      SELECT prediction_time_ms, abs_error
      FROM {table_name}
      WHERE symbol = ? AND y_true IS NOT NULL
        AND prediction_time_ms BETWEEN ? AND ?
    """
    dfp = pd.read_sql(q, conn, params=(symbol, t0, t1))
    conn.close()
    if dfp.empty:
        return pd.Series(1.0, index=train_df.index)
    dfp = dfp.dropna(subset=["abs_error"])
    if dfp.empty:
        return pd.Series(1.0, index=train_df.index)
    agg = dfp.groupby("prediction_time_ms", as_index=False)["abs_error"].max()
    mAE = np.median(agg["abs_error"].values)
    eps = max(mAE, 1e-8)
    agg["weight"] = 1.0 + alpha * (agg["abs_error"] / eps)
    agg["weight"] = agg["weight"].clip(lower=0.5, upper=max_w)
    agg["prediction_time"] = pd.to_datetime(agg["prediction_time_ms"], unit="ms")
    joined = train_df[["timestamp"]].merge(agg[["prediction_time","weight"]],
                                           left_on="timestamp", right_on="prediction_time", how="left")
    w = joined["weight"].fillna(1.0)
    w.index = train_df.index
    return w


def _delete_future_predictions(db_path: str, symbol: str, from_ms: int, table_name: str) -> int:
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute(
        f"DELETE FROM {table_name} WHERE symbol = ? AND prediction_time_ms >= ?",
        (symbol, int(from_ms)),
    )
    deleted = cur.rowcount if cur.rowcount is not None else 0
    conn.commit()
    cur.execute("PRAGMA optimize;")
    conn.commit()
    conn.close()
    return deleted


def _init_forward_state(df: pd.DataFrame):
    closes = df["close"].values
    if len(closes) < 300:
        raise ValueError("Not enough history to initialize forward state (need >=300 bars).")
    last_time = pd.to_datetime(df["timestamp"].iloc[-1])
    last_close = float(df["close"].iloc[-1])
    d7 = deque(df["close"].iloc[-7:].astype(float).tolist(), maxlen=7)
    d14 = deque(df["close"].iloc[-14:].astype(float).tolist(), maxlen=14)
    sum7 = float(np.sum(d7)); sum14 = float(np.sum(d14))
    d288 = deque(df["close"].iloc[-288:].astype(float).tolist(), maxlen=288)
    ema7 = float(df["ema_7"].iloc[-1]) if "ema_7" in df.columns else last_close
    ema14 = float(df["ema_14"].iloc[-1]) if "ema_14" in df.columns else last_close
    period = 14
    deltas = np.diff(df["close"].iloc[-(period+1):].astype(float).values)
    gains = np.clip(deltas, 0, None)
    losses = np.clip(-deltas, 0, None)
    avg_gain = float(np.mean(gains)) if len(gains) > 0 else 0.0
    avg_loss = float(np.mean(losses)) if len(losses) > 0 else 0.0
    return {"time": last_time, "close": last_close,
            "sma7_deque": d7, "sma14_deque": d14, "sum7": sum7, "sum14": sum14,
            "d288": d288, "ema7": ema7, "ema14": ema14,
            "rsi_avg_gain": avg_gain, "rsi_avg_loss": avg_loss}


def _feat_from_state(state):
    r1d = 0.0
    if len(state["d288"]) == 288:
        old = state["d288"][0]
        if old != 0:
            r1d = (state["close"] / old) - 1.0
    sma7 = state["sum7"] / 7.0
    sma14 = state["sum14"] / 14.0
    ag = state["rsi_avg_gain"]; al = state["rsi_avg_loss"]
    if al == 0 and ag == 0:
        rsi = 50.0
    elif al == 0 and ag > 0:
        rsi = 100.0
    else:
        rs = ag / al
        rsi = 100.0 - 100.0 / (1.0 + rs)
    return np.array([r1d, sma7, sma14, state["ema7"], state["ema14"], rsi], dtype=float)


def _update_state_with_pred(state, new_close):
    prev_close = state["close"]
    if len(state["sma7_deque"]) == 7:
        state["sum7"] -= state["sma7_deque"][0]
    state["sma7_deque"].append(new_close)
    state["sum7"] += new_close
    if len(state["sma14_deque"]) == 14:
        state["sum14"] -= state["sma14_deque"][0]
    state["sma14_deque"].append(new_close)
    state["sum14"] += new_close
    if len(state["d288"]) == 288:
        state["d288"].popleft()
    state["d288"].append(new_close)
    alpha7 = 2.0 / (7.0 + 1.0)
    alpha14 = 2.0 / (14.0 + 1.0)
    state["ema7"] = state["ema7"] + alpha7 * (new_close - state["ema7"])
    state["ema14"] = state["ema14"] + alpha14 * (new_close - state["ema14"])
    delta = new_close - prev_close
    gain = max(delta, 0.0);
    loss = max(-delta, 0.0)
    period = 14
    state["rsi_avg_gain"] = (state["rsi_avg_gain"] * (period - 1) + gain) / period
    state["rsi_avg_loss"] = (state["rsi_avg_loss"] * (period - 1) + loss) / period
    state["close"] = new_close
    state["time"] = state["time"] + timedelta(minutes=INTERVAL_TO_MIN["5m"])


def _predict_forward_steps(model_paths, state, steps, usage_path="ml/model_usage.json"):
    rows = []
    step_minutes = INTERVAL_TO_MIN[INTERVAL]
    for _ in range(steps):
        pred_time = state["time"]
        target_time = pred_time + timedelta(minutes=step_minutes)
        feats_vals = _feat_from_state(state)
        feats_df = pd.DataFrame([feats_vals], columns=FORWARD_FEATURE_COLS).astype(float)
        new_close = float(
            predict_weighted_prices(
                feats_df, FORWARD_FEATURE_COLS, model_paths, usage_path=usage_path
            )[0]
        )
        pred_local = pd.Timestamp(pred_time, tz="UTC").tz_convert(PRAGUE_TZ)
        target_local = pd.Timestamp(target_time, tz="UTC").tz_convert(PRAGUE_TZ)
        rows.append((
            SYMBOL, INTERVAL, FORWARD_STEPS,
            int(pd.Timestamp(pred_time).value // 1_000_000),
            int(pd.Timestamp(target_time).value // 1_000_000),
            pred_local.strftime("%Y-%m-%d %H:%M:%S"),
            target_local.strftime("%Y-%m-%d %H:%M:%S"),
            new_close,
            None, None,
            "RandomForestRegressor", FEATURES_VERSION_FWD,
            _created_at_iso()
        ))
        _update_state_with_pred(state, new_close)
    return rows


def main():
    step(1, 5, "Import latest data")
    try:
        from db.btc_import import import_latest_data
        import_latest_data()
    except Exception as exc:
        p(f"btc_import failed: {exc}")



def main():
    df = get_price_data(SYMBOL, db_path=DB_PATH)
    if df.empty:
        p("No data loaded.");
        return
    df = create_features(df)
    df = _prepare_reg_target(df, FORWARD_STEPS)
    df = df[df["target_close"].notna()].copy()
    p(f"Rows after features+target: {len(df)}")


step(3, 5, "Backfill and cleanup predictions")
create_predictions_table(DB_PATH, TABLE_PRED)
_ensure_indexes(TABLE_PRED)
backfill_actuals_and_errors(db_path=DB_PATH, table_pred=TABLE_PRED, symbol=SYMBOL)
last_ts = df["timestamp"].max()
deleted = _delete_future_predictions(DB_PATH, SYMBOL, int(last_ts.value // 1_000_000), TABLE_PRED)
p(f"  -> cleanup: deleted {deleted} future prediction rows (>= {last_ts})")

step(4, 5, "Train model on latest data")
train_start = last_ts - pd.DateOffset(years=TRAIN_WINDOW_Y)
earliest = df["timestamp"].min()
if train_start < earliest:
    train_start = earliest
latest_df = df[(df["timestamp"] >= train_start) & (df["timestamp"] <= last_ts)]
X_latest, y_latest = latest_df[FEATURE_COLS], latest_df["target_close"]
weights = _weights_from_errors_for_range(latest_df, DB_PATH, SYMBOL, TABLE_PRED, alpha=1.0, max_w=5.0)
with timed("Train latest model"):
    train_regressor(
        X_latest, y_latest,
        model_path="ml/model_reg.pkl",
        sample_weight=weights.values,
        params=dict(
            n_estimators=600,
            random_state=42,
            n_jobs=-1,
            min_samples_leaf=2,
            verbose=0,
        ),
        compress=3,
    )

model_paths = sorted(glob("ml/model_reg*.pkl"))
step(5, 5, "Predict next 3h")
state = _init_forward_state(df)
steps = int(180 / INTERVAL_TO_MIN[INTERVAL])
rows = _predict_forward_steps(model_paths, state, steps)
if rows:
    save_predictions(rows, DB_PATH, TABLE_PRED)
    p(f"Saved {len(rows)} forward rows.")
else:
    p("No rows generated.")

p("Done. Latest prices imported and 3h forecast generated.")

if __name__ == "__main__":
    main()