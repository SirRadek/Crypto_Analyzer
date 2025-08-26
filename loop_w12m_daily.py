# loop_wf12m_daily.py
# Walk-forward with a SLIDING window and forward prediction beyond last DB day.
# For each historical prediction day D (<= last DB day):
#   - Train on [D-5 years, D)
#   - Predict day D (00:00..23:59)
#   - Save predictions -> backfill -> cleanup old preds
# After the last DB day:
#   - Freeze the latest model and predict N forward days using recursive features.

from datetime import datetime, timezone, timedelta
import sqlite3
import pandas as pd
import numpy as np
from collections import deque
from zoneinfo import ZoneInfo

from db.db_connector import get_price_data
from analysis.feature_engineering import create_features, FEATURE_COLUMNS
from ml.train_regressor import train_regressor, load_regressor
from db.predictions_store import create_predictions_table, save_predictions
from analysis.compare_predictions import backfill_actuals_and_errors

# Optional progress helpers
try:
    from utils.progress import step, timed, p
except Exception:
    def p(msg): print(msg, flush=True)
    def step(i, n, msg): p(f"[{i}/{n}] {msg}")
    from contextlib import contextmanager
    @contextmanager
    def timed(label): p(f"{label} ..."); yield; p(f"{label} done")

# ---- Config ----
SYMBOL   = "BTCUSDT"
INTERVAL = "5m"
DB_PATH  = "db/data/crypto_data.sqlite"

FEATURE_COLS = FEATURE_COLUMNS
FORWARD_STEPS    = 1                         # predict close[t+1 bar]
START_DAY_STR    = "2025-08-24"             # first prediction day (inclusive)
END_DAY_STR      = None                     # None => go to last DB day (+ forward if enabled)
TRAIN_WINDOW_Y   = 5                         # sliding window length in years
TABLE_PRED       = "predictions"

# Local timezone for readability
PRAGUE_TZ = ZoneInfo("Europe/Prague")

# Forward prediction beyond last DB day
PREDICT_FUTURE_DAYS = 1                      # <<< change as you like (e.g. 1, 3, 7, 30)
FORWARD_MODE_ENABLED = True                  # set False to disable forward mode

DAY_PREDICT_START_TIME = "00:00:00"
DAY_PREDICT_END_TIME   = "23:59:59"

FEATURES_VERSION_HIST  = "wf5yD1"            # tag: wf5yD1_YYYY-MM-DD for historical days
FEATURES_VERSION_FWD   = "wf5yD1_future"     # tag: wf5yD1_future_YYYY-MM-DD for forward days

INTERVAL_TO_MIN = {
    "1m":1,"3m":3,"5m":5,"15m":15,"30m":30,"1h":60,"2h":120,"4h":240,"1d":1440
}

# --- helpers (DB/index/cleanup) ---
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

def _day_bounds(day: pd.Timestamp):
    start = pd.Timestamp(f"{day.date()} {DAY_PREDICT_START_TIME}")
    end   = pd.Timestamp(f"{day.date()} {DAY_PREDICT_END_TIME}")
    return start, end

def _created_at_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

def _weights_from_errors_for_range(train_df: pd.DataFrame,
                                   db_path: str,
                                   symbol: str,
                                   table_name: str,
                                   alpha: float = 1.0,
                                   max_w: float = 5.0) -> pd.Series:
    """
    Build sample weights for rows in train_df using backfilled abs_error from
    the predictions table.

    weight = 1 + alpha * (abs_error / median_abs_error)
    clipped to [0.5, max_w].
    """
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

    joined = train_df[["timestamp"]].merge(
        agg[["prediction_time", "weight"]],
        left_on="timestamp",
        right_on="prediction_time",
        how="left",
    )
    w = joined["weight"].fillna(1.0)
    w.index = train_df.index
    return w

def _cleanup_predictions(db_path: str, symbol: str, keep_from_ms: int, table_name: str) -> int:
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute(
        f"DELETE FROM {table_name} WHERE symbol = ? AND prediction_time_ms < ?",
        (symbol, int(keep_from_ms))
    )
    deleted = cur.rowcount if cur.rowcount is not None else 0
    conn.commit()
    cur.execute("PRAGMA optimize;")
    conn.commit()
    conn.close()
    return deleted

# --- forward feature state (EMA/SMA/RSI/return_1d) ---
def _init_forward_state(df: pd.DataFrame):
    """
    Build rolling state from the end of df.
    Assumes df already has at least 288 closes for return_1d and 14 for RSI.
    """
    closes = df["close"].values
    ts     = df["timestamp"].values
    if len(closes) < 300:
        raise ValueError("Not enough history to initialize forward state (need >= 300 bars).")

    last_time = pd.to_datetime(df["timestamp"].iloc[-1])
    last_close = float(df["close"].iloc[-1])

    # Rolling windows for SMA
    d7 = deque(df["close"].iloc[-7:].astype(float).tolist(), maxlen=7)
    d14 = deque(df["close"].iloc[-14:].astype(float).tolist(), maxlen=14)
    sum7 = float(np.sum(d7))
    sum14 = float(np.sum(d14))

    # For return_1d (288 bars back)
    d288 = deque(df["close"].iloc[-288:].astype(float).tolist(), maxlen=288)

    # EMA init from history (use feature cols if present, else compute from last window)
    if "ema_7" in df.columns and "ema_14" in df.columns:
        ema7 = float(df["ema_7"].iloc[-1])
        ema14 = float(df["ema_14"].iloc[-1])
    else:
        # simple init: current close
        ema7 = last_close
        ema14 = last_close

    # RSI state (Wilder)
    period = 14
    deltas = np.diff(df["close"].iloc[-(period+1):].astype(float).values)
    gains = np.clip(deltas, 0, None)
    losses = np.clip(-deltas, 0, None)
    avg_gain = float(np.mean(gains)) if len(gains) > 0 else 0.0
    avg_loss = float(np.mean(losses)) if len(losses) > 0 else 0.0

    return {
        "time": last_time,
        "close": last_close,
        "sma7_deque": d7, "sma14_deque": d14, "sum7": sum7, "sum14": sum14,
        "d288": d288,
        "ema7": ema7, "ema14": ema14,
        "rsi_avg_gain": avg_gain, "rsi_avg_loss": avg_loss,
    }

def _feat_from_state(state):
    # return_1d uses *current* close vs. 288 bars back
    r1d = 0.0
    if len(state["d288"]) == 288:
        old = state["d288"][0]
        if old != 0:
            r1d = (state["close"] / old) - 1.0

    sma7  = state["sum7"] / 7.0
    sma14 = state["sum14"] / 14.0

    # RSI from Wilder's averages
    ag = state["rsi_avg_gain"]; al = state["rsi_avg_loss"]
    if al == 0 and ag == 0:
        rsi = 50.0
    elif al == 0 and ag > 0:
        rsi = 100.0
    else:
        rs  = ag / al
        rsi = 100.0 - 100.0 / (1.0 + rs)

    return np.array([r1d, sma7, sma14, state["ema7"], state["ema14"], rsi], dtype=float)

def _update_state_with_pred(state, new_close):
    """
    After predicting next close, update rolling stats to be used for *next* prediction.
    """
    prev_close = state["close"]
    # Update SMA
    # Remove oldest and add new
    if len(state["sma7_deque"]) == 7:
        state["sum7"] -= state["sma7_deque"][0]
    state["sma7_deque"].append(new_close)
    state["sum7"] += new_close

    if len(state["sma14_deque"]) == 14:
        state["sum14"] -= state["sma14_deque"][0]
    state["sma14_deque"].append(new_close)
    state["sum14"] += new_close

    # Update 288 deque for return_1d
    if len(state["d288"]) == 288:
        state["d288"].popleft()
    state["d288"].append(new_close)

    # EMA updates
    alpha7 = 2.0 / (7.0 + 1.0)
    alpha14 = 2.0 / (14.0 + 1.0)
    state["ema7"] = state["ema7"] + alpha7 * (new_close - state["ema7"])
    state["ema14"] = state["ema14"] + alpha14 * (new_close - state["ema14"])

    # RSI Wilder update
    delta = new_close - prev_close
    gain = max(delta, 0.0)
    loss = max(-delta, 0.0)
    period = 14
    state["rsi_avg_gain"] = (state["rsi_avg_gain"] * (period - 1) + gain) / period
    state["rsi_avg_loss"] = (state["rsi_avg_loss"] * (period - 1) + loss) / period

    # advance time/close by 5 minutes
    state["close"] = new_close
    state["time"] = state["time"] + timedelta(minutes=INTERVAL_TO_MIN["5m"])

def _predict_forward_day(model, state, day_start, day_end):
    """
    Produce 5m ahead predictions for one full forward day.
    We generate rows whose TARGET time is within [day_start, day_end].
    """
    rows = []
    # We will step until target_time reaches day_end (inclusive)
    step_minutes = INTERVAL_TO_MIN["5m"]
    while True:
        pred_time = state["time"]            # features at 'pred_time' (known state)
        target_time = pred_time + timedelta(minutes=step_minutes)
        if target_time > day_end:
            break
        if target_time < day_start:
            # advance state with a predicted step but don't record if target is before day_start
            feats_vals = _feat_from_state(state)  # shape (n_features,)
            feats_df = pd.DataFrame([feats_vals], columns=FEATURE_COLS).astype(float)
            new_close = float(model.predict(feats_df)[0])
            _update_state_with_pred(state, new_close)
            continue

        feats_vals = _feat_from_state(state)  # shape (n_features,)
        feats_df = pd.DataFrame([feats_vals], columns=FEATURE_COLS).astype(float)
        new_close = float(model.predict(feats_df)[0])

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
            "RandomForestRegressor", f"{FEATURES_VERSION_FWD}_{day_start.date()}",
            datetime.now(timezone.utc).isoformat()
        ))

        _update_state_with_pred(state, new_close)

    return rows

def main():
    step(1, 10, f"Load full series for {SYMBOL}")
    with timed("Load + features + target"):
        df = get_price_data(SYMBOL, db_path=DB_PATH)
        if df.empty:
            p("No data loaded."); return
        df = create_features(df)
        df = _prepare_reg_target(df, FORWARD_STEPS)
        df = df[df["target_close"].notna()].copy()
        p(f"Rows after features+target: {len(df)}")

    # Day range
    first_day = df["timestamp"].min().floor("D")
    last_db_day  = df["timestamp"].max().floor("D")
    pred_day  = pd.to_datetime(START_DAY_STR).floor("D")
    if pred_day < first_day:
        p(f"Adjusting start from {pred_day.date()} to first data day {first_day.date()}.")
        pred_day = first_day

    # Historical part end
    hist_end_day = (pd.to_datetime(END_DAY_STR).floor("D") if END_DAY_STR else last_db_day)

    # If forward is enabled and END not given, extend to future
    if FORWARD_MODE_ENABLED and END_DAY_STR is None:
        fwd_end = last_db_day + pd.Timedelta(days=PREDICT_FUTURE_DAYS)
        overall_end_day = max(hist_end_day, fwd_end)
    else:
        overall_end_day = hist_end_day

    step(2, 10, f"Prepare table '{TABLE_PRED}' in {DB_PATH}")
    create_predictions_table(DB_PATH, TABLE_PRED)
    _ensure_indexes(TABLE_PRED)

    horizon_minutes = INTERVAL_TO_MIN[INTERVAL] * FORWARD_STEPS
    day_idx = 0
    total_days = (overall_end_day - pred_day).days + 1
    p(f"Loop days: {total_days} ({pred_day.date()} → {overall_end_day.date()})")

    latest_trained_model = None

    # ------ HISTORICAL LOOP (<= last_db_day) ------
    while pred_day <= hist_end_day:
        day_idx += 1
        features_version = f"{FEATURES_VERSION_HIST}_{pred_day.date()}"
        step(3, 10, f"[{day_idx}/{total_days}] Historical day {pred_day.date()}")

        # Training window: [pred_day - TRAIN_WINDOW_Y years, pred_day)
        day_start, day_end = _day_bounds(pred_day)
        train_start = day_start - pd.DateOffset(years=TRAIN_WINDOW_Y)
        earliest = df["timestamp"].min()
        if train_start < earliest:
            train_start = earliest

        train_df = df[(df["timestamp"] >= train_start) & (df["timestamp"] < day_start)]
        pred_df  = df[(df["timestamp"] >= day_start) & (df["timestamp"] <= day_end)].copy()
        p(f"  -> train rows={len(train_df)}, predict rows={len(pred_df)} "
          f"(train window: {train_start} .. {day_start - pd.Timedelta(seconds=1)})")

        if train_df.empty or pred_df.empty:
            p("  (skip) Not enough data for this day.")
            pred_day += pd.Timedelta(days=1)
            continue

        X_train, y_train = train_df[FEATURE_COLS], train_df["target_close"]

        # optional: weights from past errors within window (can be left as ones)
        weights = _weights_from_errors_for_range(train_df, DB_PATH, SYMBOL, TABLE_PRED,
                                                 alpha=1.0, max_w=5.0)
        p(f"  -> weighted rows (w!=1): {(weights!=1).sum()} | "
          f"median_w={np.median(weights):.3f} max_w={weights.max():.3f}")

        with timed(f"Train for {pred_day.date()}"):
            latest_trained_model = train_regressor(
                X_train, y_train,
                model_path="ml/model_reg.pkl",
                sample_weight=weights.values,
                params=dict(
                    n_estimators=600,
                    random_state=42,
                    n_jobs=-1,
                    min_samples_leaf=2,
                    verbose=0
                ),
                compress=3
            )

        with timed(f"Predict {pred_day.date()}"):
            Xp_df = pred_df[FEATURE_COLS].astype(float)  # keep column names
            y_pred = latest_trained_model.predict(Xp_df)

        pred_df["prediction_time_ms"] = (pred_df["timestamp"].astype("int64") // 10**6)
        target_dt = pred_df["timestamp"] + pd.Timedelta(minutes=horizon_minutes)
        pred_df["target_time_ms"] = (target_dt.astype("int64") // 10**6)

        created_at = _created_at_iso()
        rows = []
        for j, yp in enumerate(y_pred):
            pred_local = pred_df.iloc[j]["timestamp"].tz_localize("UTC").tz_convert(PRAGUE_TZ)
            target_local = target_dt.iloc[j].tz_localize("UTC").tz_convert(PRAGUE_TZ)
            rows.append((
                SYMBOL, INTERVAL, FORWARD_STEPS,
                int(pred_df.iloc[j]["prediction_time_ms"]),
                int(pred_df.iloc[j]["target_time_ms"]),
                pred_local.strftime("%Y-%m-%d %H:%M:%S"),
                target_local.strftime("%Y-%m-%d %H:%M:%S"),
                float(yp),
                None, None,
                "RandomForestRegressor", features_version,
                created_at
            ))

        with timed(f"Save {len(rows)} rows for {features_version}"):
            save_predictions(rows, DB_PATH, TABLE_PRED)

        with timed(f"Backfill {features_version}"):
            backfill_actuals_and_errors(
                db_path=DB_PATH,
                table_pred=TABLE_PRED,
                symbol=SYMBOL
            )

        # Cleanup old predictions (older than next day's train window)
        next_day_start = day_start + pd.Timedelta(days=1)
        next_train_start = next_day_start - pd.DateOffset(years=TRAIN_WINDOW_Y)
        keep_from_ms = int(next_train_start.value // 1_000_000)
        deleted = _cleanup_predictions(DB_PATH, SYMBOL, keep_from_ms, TABLE_PRED)
        p(f"  -> cleanup: deleted {deleted} old prediction rows (< {next_train_start})")

        if day_idx % 30 == 0:
            conn = sqlite3.connect(DB_PATH)
            conn.execute("VACUUM;")
            conn.close()
            p("  -> VACUUM performed (monthly)")

        pred_day += pd.Timedelta(days=1)

    # ------ FORWARD LOOP (> last_db_day) ------
    if FORWARD_MODE_ENABLED and pred_day <= overall_end_day:
        step(9, 10, f"Forward mode: predicting beyond last DB day ({last_db_day.date()})")
        # Ensure we have a model (train on the last historical window if needed)
        if latest_trained_model is None:
            # train once on the last available sliding window ending at last_db_day
            last_day_start, _ = _day_bounds(last_db_day)
            train_start = last_day_start - pd.DateOffset(years=TRAIN_WINDOW_Y)
            earliest = df["timestamp"].min()
            if train_start < earliest:
                train_start = earliest
            base_train_df = df[(df["timestamp"] >= train_start) & (df["timestamp"] < last_day_start)]
            X_train, y_train = base_train_df[FEATURE_COLS], base_train_df["target_close"]
            latest_trained_model = train_regressor(
                X_train, y_train,
                model_path="ml/model_reg.pkl",
                sample_weight=None,
                params=dict(
                    n_estimators=600,
                    random_state=42,
                    n_jobs=-1,
                    min_samples_leaf=2,
                    verbose=0
                ),
                compress=3
            )

        # Initialize forward state from the tail of df
        base_state = _init_forward_state(df)

        while pred_day <= overall_end_day:
            day_start, day_end = _day_bounds(pred_day)
            step(10, 10, f"Forward day {pred_day.date()} (no backfill yet)")
            with timed(f"Forward predict {pred_day.date()}"):
                rows = _predict_forward_day(latest_trained_model, base_state, day_start, day_end)
            if rows:
                with timed(f"Save {len(rows)} forward rows for {pred_day.date()}"):
                    save_predictions(rows, DB_PATH, TABLE_PRED)
            else:
                p("  -> no rows generated (check intervals/history).")
            # (no backfill/cleanup in forward mode – truth not available yet)
            pred_day += pd.Timedelta(days=1)

    p("Done. Historical days backfilled; future days queued with y_true=NULL.")
    p("When new prices are imported, run:  python backfill_compare.py  to fill y_true/abs_error.")

if __name__ == "__main__":
    main()
