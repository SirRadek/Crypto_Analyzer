# loop_forward_12h.py
# Train on full historical data and predict the next 12 hours using recursive features.

import sqlite3
from collections import deque
from datetime import datetime, timezone, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from analysis.compare_predictions import backfill_actuals_and_errors
from analysis.feature_engineering import create_features
from db.db_connector import get_price_data
from db.predictions_store import create_predictions_table, save_predictions
from ml.train_regressor import train_regressor

# Optional progress helpers
try:
    from utils.progress import step, timed, p
except Exception:
    def p(msg):
        print(msg, flush=True)
    def step(i, n, msg):
        p(f"[{i}/{n}] {msg}")
    from contextlib import contextmanager
    @contextmanager
    def timed(label):
        p(f"{label} ...")
        yield
        p(f"{label} done")

# ---- Config ----
SYMBOL = "BTCUSDT"
INTERVAL = "5m"
DB_PATH = "db/data/crypto_data.sqlite"

# Features that can be rolled forward from price-only information
FEATURE_COLS = [
    "return_1d",
    "sma_7",
    "sma_14",
    "ema_7",
    "ema_14",
    "rsi_14",
]
FORWARD_STEPS = 1                 # predict close[t+1 bar]
FORWARD_HOURS = 12                # generate predictions for next 12 hours
TABLE_PRED = "predictions"
FEATURES_VERSION_FWD = "full12h"

# Local timezone for readability
PRAGUE_TZ = ZoneInfo("Europe/Prague")

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

# --- helpers ---
def _ensure_indexes(table_name: str):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("CREATE INDEX IF NOT EXISTS idx_prices_open_time ON prices(open_time)")
    cur.execute(
        f"CREATE INDEX IF NOT EXISTS idx_{table_name}_pred_time ON {table_name}(prediction_time_ms)"
    )
    cur.execute(
        f"CREATE INDEX IF NOT EXISTS idx_{table_name}_target_time ON {table_name}(target_time_ms)"
    )
    cur.execute(f"CREATE INDEX IF NOT EXISTS idx_{table_name}_symbol ON {table_name}(symbol)")
    conn.commit()
    conn.close()


def _prepare_reg_target(df: pd.DataFrame, forward_steps: int) -> pd.DataFrame:
    df = df.copy()
    df["target_close"] = df["close"].shift(-forward_steps)
    return df


# --- forward feature state ---
def _init_forward_state(df: pd.DataFrame):
    """Build rolling state from the end of df."""
    closes = df["close"].values
    if len(closes) < 300:
        raise ValueError("Not enough history to initialize forward state (need >= 300 bars).")

    last_time = pd.to_datetime(df["timestamp"].iloc[-1])
    last_close = float(df["close"].iloc[-1])

    d7 = deque(df["close"].iloc[-7:].astype(float).tolist(), maxlen=7)
    d14 = deque(df["close"].iloc[-14:].astype(float).tolist(), maxlen=14)
    sum7 = float(np.sum(d7))
    sum14 = float(np.sum(d14))

    d288 = deque(df["close"].iloc[-288:].astype(float).tolist(), maxlen=288)

    if "ema_7" in df.columns and "ema_14" in df.columns:
        ema7 = float(df["ema_7"].iloc[-1])
        ema14 = float(df["ema_14"].iloc[-1])
    else:
        ema7 = last_close
        ema14 = last_close

    period = 14
    deltas = np.diff(df["close"].iloc[-(period + 1):].astype(float).values)
    gains = np.clip(deltas, 0, None)
    losses = np.clip(-deltas, 0, None)
    avg_gain = float(np.mean(gains)) if len(gains) > 0 else 0.0
    avg_loss = float(np.mean(losses)) if len(losses) > 0 else 0.0

    return {
        "time": last_time,
        "close": last_close,
        "sma7_deque": d7,
        "sma14_deque": d14,
        "sum7": sum7,
        "sum14": sum14,
        "d288": d288,
        "ema7": ema7,
        "ema14": ema14,
        "rsi_avg_gain": avg_gain,
        "rsi_avg_loss": avg_loss,
    }


def _feat_from_state(state):
    r1d = 0.0
    if len(state["d288"]) == 288:
        old = state["d288"][0]
        if old != 0:
            r1d = (state["close"] / old) - 1.0

    sma7 = state["sum7"] / 7.0
    sma14 = state["sum14"] / 14.0

    ag = state["rsi_avg_gain"]
    al = state["rsi_avg_loss"]
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
    gain = max(delta, 0.0)
    loss = max(-delta, 0.0)
    period = 14
    state["rsi_avg_gain"] = (state["rsi_avg_gain"] * (period - 1) + gain) / period
    state["rsi_avg_loss"] = (state["rsi_avg_loss"] * (period - 1) + loss) / period

    state["close"] = new_close
    state["time"] = state["time"] + timedelta(minutes=INTERVAL_TO_MIN["5m"])


def _predict_forward_hours(model, state, hours):
    rows = []
    step_minutes = INTERVAL_TO_MIN[INTERVAL]
    steps = int(hours * 60 / step_minutes)
    for _ in range(steps):
        next_time = state["time"] + timedelta(minutes=step_minutes)

        feats_vals = _feat_from_state(state)
        feats_df = pd.DataFrame([feats_vals], columns=FEATURE_COLS).astype(float)
        new_close = float(model.predict(feats_df)[0])

        pred_local = pd.Timestamp(next_time, tz="UTC").tz_convert(PRAGUE_TZ)
        target_local = pred_local
        rows.append(
            (
                SYMBOL,
                INTERVAL,
                FORWARD_STEPS,
                int(pd.Timestamp(next_time).value // 1_000_000),
                int(pd.Timestamp(next_time).value // 1_000_000),
                pred_local.strftime("%Y-%m-%d %H:%M:%S"),
                target_local.strftime("%Y-%m-%d %H:%M:%S"),
                new_close,
                None,
                None,
                "RandomForestRegressor",
                FEATURES_VERSION_FWD,
                datetime.now(timezone.utc).isoformat(),
            )
        )

        _update_state_with_pred(state, new_close)

    return rows


def main():
    step(1, 5, "Backfill previous predictions")
    create_predictions_table(DB_PATH, TABLE_PRED)
    _ensure_indexes(TABLE_PRED)
    with timed("Backfill"):
        backfill_actuals_and_errors(db_path=DB_PATH, table_pred=TABLE_PRED, symbol=SYMBOL)

    step(2, 5, f"Load full series for {SYMBOL}")
    with timed("Load + features"):
        df = get_price_data(SYMBOL, db_path=DB_PATH)
        df = create_features(df)
        # Ensure return_1d reflects the change over one day (288 bars)
        df["return_1d"] = df["close"].pct_change(288)
        df_train = _prepare_reg_target(df, FORWARD_STEPS)
        df_train = df_train[df_train["target_close"].notna()].copy()

    step(3, 5, "Train model on all data")
    X_train, y_train = df_train[FEATURE_COLS], df_train["target_close"]
    with timed("Train"):
        model = train_regressor(
            X_train,
            y_train,
            model_path="ml/model_reg.pkl",
            sample_weight=None,
            params=dict(
                n_estimators=600,
                random_state=42,
                n_jobs=-1,
                min_samples_leaf=2,
                verbose=0,
            ),
            compress=3,
        )

    step(4, 5, "Predict next 12 hours")
    state = _init_forward_state(df)
    with timed("Forward predict"):
        rows = _predict_forward_hours(model, state, FORWARD_HOURS)

    step(5, 5, f"Save {len(rows)} prediction rows")
    with timed("Save"):
        save_predictions(rows, DB_PATH, TABLE_PRED)

    p("Done.")


if __name__ == "__main__":
    main()