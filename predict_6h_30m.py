from datetime import timedelta, datetime, timezone
import time, sqlite3
import pandas as pd
import numpy as np
from collections import deque

from db.db_connector import get_price_data
from analysis.feature_engineering import create_features
from ml.train_regressor import train_regressor
from db.predictions_store import create_predictions_table, save_predictions
from db import btc_import
from analysis.compare_predictions import backfill_actuals_and_errors

SYMBOL = "BTCUSDT"
DB_PATH = "db/data/crypto_data.sqlite"

FEATURE_COLS = ['return_1d', 'sma_7', 'sma_14', 'ema_7', 'ema_14', 'rsi_14']
FORWARD_STEPS = 1                      # model predicts close[t+1 bar]
TRAIN_WINDOW_Y = 1                     # two-year training window
PREDICT_HOURS = 12                      # forecast horizon
STEP_MIN = 5                           # underlying data interval (minutes)
OUTPUT_INTERVAL_MIN = 15               # output every 30 minutes


def _update_prices():
    """Fetch latest price data via btc_import up to current time."""
    btc_import.create_db()
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT MAX(open_time) FROM prices WHERE symbol = ?", (SYMBOL,))
    row = cur.fetchone()
    conn.close()
    last_ts = row[0] if row and row[0] else btc_import.date_to_milliseconds("2020-01-01")
    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    start_ts = last_ts + 1
    while start_ts < now_ms:
        klines = btc_import.get_klines(SYMBOL, f"{STEP_MIN}m", start_ts, now_ms)
        if not klines:
            break
        btc_import.save_to_db(klines, SYMBOL, f"{STEP_MIN}m")
        start_ts = klines[-1][0] + 1
        time.sleep(0.4)


def _load_backfilled_errors(db_path: str, symbol: str, table_name: str) -> pd.DataFrame:
    conn = sqlite3.connect(db_path)
    q = f"""
        SELECT prediction_time_ms, target_time_ms, y_pred, y_true, abs_error
        FROM {table_name}
        WHERE symbol = ? AND y_true IS NOT NULL
    """
    dfp = pd.read_sql(q, conn, params=(symbol,))
    conn.close()
    if dfp.empty:
        return dfp
    dfp["prediction_time"] = pd.to_datetime(dfp["prediction_time_ms"], unit="ms")
    return dfp


def _build_sample_weights(base_df: pd.DataFrame, preds_df: pd.DataFrame,
                          alpha: float = 1.0, max_w: float = 5.0) -> pd.Series:
    w = pd.Series(1.0, index=base_df.index)
    if preds_df.empty:
        return w
    mAE = np.median(preds_df["abs_error"].values) if len(preds_df) else 0.0
    eps = max(mAE, 1e-8)
    preds_df = preds_df.copy()
    preds_df["weight"] = 1.0 + alpha * (preds_df["abs_error"] / eps)
    preds_df["weight"] = preds_df["weight"].clip(lower=0.5, upper=max_w)
    # Multiple predictions can share the same prediction_time (e.g. reruns or
    # different horizons). Pandas' map requires a unique index so we aggregate
    # duplicate timestamps by averaging their weights.
    weight_map = preds_df.groupby("prediction_time")["weight"].mean()
    aligned_idx = base_df["timestamp"].map(weight_map)
    w.loc[aligned_idx.notna().values] = aligned_idx.dropna().values
    return w


def _prepare_reg_target(df: pd.DataFrame, forward_steps: int) -> pd.DataFrame:
    df = df.copy()
    df["target_close"] = df["close"].shift(-forward_steps)
    return df


def _init_forward_state(df: pd.DataFrame):
    """Initialize rolling state from the end of df."""
    closes = df["close"].values
    if len(closes) < 300:
        raise ValueError("Not enough history to initialize forward state (need >=300 bars)")

    last_time = pd.to_datetime(df["timestamp"].iloc[-1])
    last_close = float(df["close"].iloc[-1])

    d7 = deque(df["close"].iloc[-7:].astype(float).tolist(), maxlen=7)
    d14 = deque(df["close"].iloc[-14:].astype(float).tolist(), maxlen=14)
    sum7 = float(np.sum(d7))
    sum14 = float(np.sum(d14))

    d288 = deque(df["close"].iloc[-288:].astype(float).tolist(), maxlen=288)

    ema7 = float(df["ema_7"].iloc[-1]) if "ema_7" in df.columns else last_close
    ema14 = float(df["ema_14"].iloc[-1]) if "ema_14" in df.columns else last_close

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
    state["time"] = state["time"] + timedelta(minutes=STEP_MIN)


def _predict_forward(model, df):
    state = _init_forward_state(df)
    steps = int(PREDICT_HOURS * 60 / STEP_MIN)
    record_every = OUTPUT_INTERVAL_MIN // STEP_MIN
    preds = []
    for step in range(1, steps + 1):
        pred_time = state["time"]
        feats = _feat_from_state(state)
        feats_df = pd.DataFrame([feats], columns=FEATURE_COLS).astype(float)
        new_close = float(model.predict(feats_df)[0])
        _update_state_with_pred(state, new_close)
        if step % record_every == 0:
            preds.append((pred_time, state["time"], new_close))
    return preds


def main():
    _update_prices()
    backfill_actuals_and_errors(db_path=DB_PATH, table_pred="predictions", symbol=SYMBOL)

    dfp = _load_backfilled_errors(DB_PATH, SYMBOL, "predictions")
    if not dfp.empty:
        print(f"Historical MAE: {dfp['abs_error'].mean():.2f}")

    df = get_price_data(SYMBOL, db_path=DB_PATH)
    if df.empty:
        print("No data loaded.")
        return
    df = create_features(df)
    df = _prepare_reg_target(df, FORWARD_STEPS)
    df = df[df["target_close"].notna()].copy()

    last_time = df["timestamp"].max()
    train_start = last_time - pd.DateOffset(years=TRAIN_WINDOW_Y)
    train_df = df[df["timestamp"] >= train_start]
    X_train, y_train = train_df[FEATURE_COLS], train_df["target_close"]

    weights = _build_sample_weights(train_df, dfp)

    runs = []
    for i in range(5):
        params = dict(
            n_estimators=600,
            random_state=42 + i,
            n_jobs=-1,
            min_samples_leaf=2,
            verbose=0,
        )
        model = train_regressor(
            X_train,
            y_train,
            sample_weight=weights,
            model_path=f"ml/model_reg_run{i}.pkl",
            params=params,
        )
        runs.append(_predict_forward(model, df))

    avg_predictions = []
    for idx in range(len(runs[0])):
        pred_time = runs[0][idx][0]
        target_time = runs[0][idx][1]
        vals = [r[idx][2] for r in runs]
        avg_price = float(np.mean(vals))
        min_p = float(np.min(vals))
        max_p = float(np.max(vals))
        avg_predictions.append((pred_time, target_time, avg_price, min_p, max_p))

    create_predictions_table(DB_PATH, "predictions")
    rows = []
    created_at = datetime.now(timezone.utc).isoformat()
    for pred_time, target_time, price, low, high in avg_predictions:
        rows.append(
            (
                SYMBOL,
                f"{STEP_MIN}m",
                FORWARD_STEPS,
                int(pred_time.value // 1_000_000),
                int(target_time.value // 1_000_000),
                price,
                low,
                high,
                None,
                None,
                "rf_avg5",
                "predict_6h_30m_v2",
                created_at,
            )
        )
    save_predictions(rows, DB_PATH, "predictions")

    for _, target_time, price, low, high in avg_predictions:
        print(target_time, price, low, high)


if __name__ == "__main__":
    main()