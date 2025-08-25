# loop_wf6m_daily.py
# Walk-forward training with a SLIDING 6-MONTH window, starting 2020-08-08:
#   For each prediction day D:
#     - Train on [D-6 months, D)
#     - Predict the whole day D (00:00..23:59)
#     - Save predictions to 'prediction' (same DB as prices)
#     - Backfill y_true + abs_error
#     - Learn from errors via sample_weight in the next day (within its 6M window)
#     - Delete old predictions that will never be used again (keep only those
#       inside the next day's 6-month training window)
#
from datetime import datetime, timezone
import sqlite3
import pandas as pd
import numpy as np

from db.db_connector import get_price_data
from analysis.feature_engineering import create_features
from ml.train_regressor import train_regressor
from ml.predict_regressor import predict_prices
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

FEATURE_COLS = ['return_1d','sma_7','sma_14','ema_7','ema_14','rsi_14']
FORWARD_STEPS    = 1                         # predict close[t+1 bar]
START_DAY_STR    = "2025-08-23"             # first prediction day (inclusive)
END_DAY_STR      = None                     # None => run until last full day in data
TRAIN_WINDOW_M   = 12                        # 6-month sliding window

DAY_PREDICT_START_TIME = "00:00:00"
DAY_PREDICT_END_TIME   = "23:59:59"

FEATURES_VERSION_PREFIX = "wf6mD1"          # version tag: wf6mD1_YYYY-MM-DD

INTERVAL_TO_MIN = {
    "1m":1,"3m":3,"5m":5,"15m":15,"30m":30,"1h":60,"2h":120,"4h":240,"1d":1440
}

# --- helpers ---
def _ensure_indexes():
    conn = sqlite3.connect(DB_PATH); cur = conn.cursor()
    cur.execute("CREATE INDEX IF NOT EXISTS idx_prices_open_time ON prices(open_time)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_prediction_pred_time ON prediction(prediction_time_ms)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_prediction_target_time ON prediction(target_time_ms)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_prediction_symbol ON prediction(symbol)")
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
                                   alpha: float = 1.0,
                                   max_w: float = 5.0) -> pd.Series:
    """
    Build sample weights for rows in train_df using backfilled abs_error from 'prediction'
    whose prediction_time_ms falls within train_df time range.

    Strategy:
      - aggregate duplicates per timestamp by MAX(abs_error)  (most conservative)
      - weight = 1 + alpha * (abs_error / median_abs_error), clipped to [0.5, max_w]
      - merge (not map) to avoid 'unique index' requirement
    """
    # default: all ones
    if train_df.empty:
        return pd.Series(1.0, index=train_df.index)

    t0 = int(train_df["timestamp"].min().value // 1_000_000)
    t1 = int(train_df["timestamp"].max().value // 1_000_000)

    import sqlite3
    conn = sqlite3.connect(db_path)
    q = """
      SELECT prediction_time_ms, abs_error
      FROM prediction
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

    # aggregate duplicates per timestamp
    agg = dfp.groupby("prediction_time_ms", as_index=False)["abs_error"].max()
    # robust scale by median AE
    mAE = np.median(agg["abs_error"].values)
    eps = max(mAE, 1e-8)
    agg["weight"] = 1.0 + alpha * (agg["abs_error"] / eps)
    agg["weight"] = agg["weight"].clip(lower=0.5, upper=max_w)

    # to datetime for joining
    agg["prediction_time"] = pd.to_datetime(agg["prediction_time_ms"], unit="ms")

    # left-merge onto train_df timestamps
    joined = train_df[["timestamp"]].merge(
        agg[["prediction_time", "weight"]],
        left_on="timestamp",
        right_on="prediction_time",
        how="left"
    )
    w = joined["weight"].fillna(1.0)
    w.index = train_df.index  # keep original row order/index
    return w


def _cleanup_predictions(db_path: str, symbol: str, keep_from_ms: int) -> int:
    """
    Delete predictions older than keep_from_ms (epoch ms) for given symbol.
    Returns number of deleted rows.
    """
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute(
        "DELETE FROM prediction WHERE symbol = ? AND prediction_time_ms < ?",
        (symbol, int(keep_from_ms))
    )
    deleted = cur.rowcount if cur.rowcount is not None else 0
    conn.commit()
    cur.execute("PRAGMA optimize;")
    conn.commit()
    conn.close()
    return deleted

def main():
    step(1, 8, f"Load full series for {SYMBOL}")
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
    last_day  = df["timestamp"].max().floor("D")
    pred_day  = pd.to_datetime(START_DAY_STR).floor("D")
    if pred_day < first_day:
        p(f"Adjusting start from {pred_day.date()} to first data day {first_day.date()}.")
        pred_day = first_day
    end_day = (pd.to_datetime(END_DAY_STR).floor("D") if END_DAY_STR else last_day)

    step(2, 8, f"Prepare table 'prediction' in {DB_PATH}")
    create_predictions_table(DB_PATH, "prediction")
    _ensure_indexes()

    horizon_minutes = INTERVAL_TO_MIN[INTERVAL] * FORWARD_STEPS
    day_idx = 0
    total_days = (end_day - pred_day).days + 1
    p(f"Loop days: {total_days} ({pred_day.date()} → {end_day.date()})")

    # Main daily loop
    while pred_day <= end_day:
        day_idx += 1
        features_version = f"{FEATURES_VERSION_PREFIX}_{pred_day.date()}"
        step(3, 8, f"[{day_idx}/{total_days}] Predicting day {pred_day.date()} with 6-month training")

        # Training window: [pred_day - 6 months, pred_day)
        day_start, day_end = _day_bounds(pred_day)
        train_start = day_start - pd.DateOffset(months=TRAIN_WINDOW_M)
        earliest = df["timestamp"].min()
        if train_start < earliest:
            train_start = earliest

        train_df = df[(df["timestamp"] >= train_start) & (df["timestamp"] < day_start)]
        pred_df  = df[(df["timestamp"] >= day_start) & (df["timestamp"] <= day_end)].copy()
        p(f"  -> train rows={len(train_df)}, predict rows={len(pred_df)} "
          f"(train window: {train_start} .. {day_start - pd.Timedelta(seconds=1)})")

        if train_df.empty or pred_df.empty:
            p("  (skip) Not enough data for this day.")
            # still move forward by a day
            pred_day += pd.Timedelta(days=1)
            continue

        X_train, y_train = train_df[FEATURE_COLS], train_df["target_close"]

        # Build error-based sample weights from already backfilled predictions inside the window
        weights = _weights_from_errors_for_range(train_df, DB_PATH, SYMBOL,
                                                 alpha=1.0, max_w=5.0)
        p(f"  -> weighted rows (w!=1): {(weights!=1).sum()} | "
          f"median_w={np.median(weights):.3f} max_w={weights.max():.3f}")

        # Train (train_regressor enforces <=200GB via compression/auto-fallback)
        with timed(f"Train for {pred_day.date()}"):
            train_regressor(
                X_train, y_train,
                model_path="ml/model_reg.pkl",
                sample_weight=weights.values,
                params=dict(
                    n_estimators=600,  # will auto-fallback if file too large
                    random_state=42,
                    n_jobs=-1,
                    min_samples_leaf=2,
                    verbose=0
                ),
                compress=3
            )

        # Predict 1 day ahead (the day 'pred_day')
        with timed(f"Predict {pred_day.date()}"):
            y_pred = predict_prices(pred_df, FEATURE_COLS)

        # Save predictions for the day
        pred_df["prediction_time_ms"] = (pred_df["timestamp"].astype("int64") // 10**6)
        target_dt = pred_df["timestamp"] + pd.Timedelta(minutes=horizon_minutes)
        pred_df["target_time_ms"] = (target_dt.astype("int64") // 10**6)

        created_at = _created_at_iso()
        rows = []
        for j, yp in enumerate(y_pred):
            rows.append((
                SYMBOL, INTERVAL, FORWARD_STEPS,
                int(pred_df.iloc[j]["prediction_time_ms"]),
                int(pred_df.iloc[j]["target_time_ms"]),
                float(yp),
                None, None,
                "RandomForestRegressor", features_version,
                created_at
            ))

        with timed(f"Save {len(rows)} rows for {features_version}"):
            save_predictions(rows, DB_PATH, "prediction")

        # Backfill immediately (historical data ⇒ truth available)
        with timed(f"Backfill {features_version}"):
            backfill_actuals_and_errors(
                db_path=DB_PATH,
                table_pred="prediction",
                symbol=SYMBOL
            )

        # Cleanup: remove predictions older than next day's 6M training window
        next_day_start = day_start + pd.Timedelta(days=1)
        next_train_start = next_day_start - pd.DateOffset(months=TRAIN_WINDOW_M)
        keep_from_ms = int(next_train_start.value // 1_000_000)
        deleted = _cleanup_predictions(DB_PATH, SYMBOL, keep_from_ms)
        p(f"  -> cleanup: deleted {deleted} old prediction rows (< {next_train_start})")

        # Optional: occasionally reclaim disk space (expensive op)
        if day_idx % 30 == 0:
            conn = sqlite3.connect(DB_PATH)
            conn.execute("VACUUM;")
            conn.close()
            p("  -> VACUUM performed (monthly)")

        # Move to the next day
        pred_day += pd.Timedelta(days=1)

    step(8, 8, "All done — daily loop finished")
    p("You can evaluate per-day via features_version (wf6mD1_YYYY-MM-DD).")

if __name__ == "__main__":
    main()
