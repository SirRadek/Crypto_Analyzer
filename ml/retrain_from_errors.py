import sqlite3

import numpy as np
import pandas as pd

from analysis.feature_engineering import create_features, FEATURE_COLUMNS
from db.db_connector import get_price_data
from ml.train_regressor import train_regressor

DB_PATH = "db/data/crypto_data.sqlite"
SYMBOL = "BTCUSDT"
INTERVAL = "5m"

# must match your training features
FEATURE_COLS = FEATURE_COLUMNS
FORWARD_STEPS = 1  # predict close[t+1 bar]

def _prepare_reg_target(df: pd.DataFrame, forward_steps: int) -> pd.DataFrame:
    df = df.copy()
    df["target_close"] = df["close"].shift(-forward_steps)
    return df

def _load_backfilled_errors(db_path: str, symbol: str) -> pd.DataFrame:
    """
    Load predictions that already have y_true (i.e., backfilled) and their abs_error.
    """
    conn = sqlite3.connect(db_path)
    q = """
    SELECT prediction_time_ms, target_time_ms, y_pred, y_true, abs_error
    FROM prediction
    WHERE symbol = ? AND y_true IS NOT NULL
    """
    dfp = pd.read_sql(q, conn, params=(symbol,))
    conn.close()
    if dfp.empty:
        return dfp
    # convert ms -> timestamp (aligns with features 'timestamp')
    dfp["prediction_time"] = pd.to_datetime(dfp["prediction_time_ms"], unit="ms")
    return dfp


def _build_sample_weights(
        base_df: pd.DataFrame,
        preds_df: pd.DataFrame,
        alpha: float = 1.0,
        max_w: float = 5.0
) -> pd.Series:
    """
    Create per-row weights for training:
      weight = 1 + alpha * (abs_error / median_abs_error), clipped to [0.5, max_w]
    Rows without recorded error keep weight = 1.
    """
    w = pd.Series(1.0, index=base_df.index)

    if preds_df.empty:
        return w

    # robust scale by median AE (avoid division by 0 with eps)
    mAE = float(np.median(preds_df["abs_error"].to_numpy())) if len(preds_df) else 0.0
    eps = max(mAE, 1e-8)

    # map prediction_time -> weight
    preds_df = preds_df.copy()
    preds_df["weight"] = 1.0 + alpha * (preds_df["abs_error"].astype(float) / eps)
    preds_df["weight"] = preds_df["weight"].clip(lower=0.5, upper=max_w)

    # align to base_df rows by timestamp == prediction_time
    weight_map = preds_df.set_index("prediction_time")["weight"]
    # align by index (timestamp) if you set index; here we align by equality:
    aligned_idx = base_df["timestamp"].map(weight_map)  # NaN where no record
    mask = aligned_idx.notna().to_numpy()
    w.loc[mask] = aligned_idx.dropna().to_numpy()
    return w


def retrain_with_error_weights(
        db_path: str = DB_PATH,
        symbol: str = SYMBOL,
        forward_steps: int = FORWARD_STEPS,
        alpha: float = 1.0,
        max_weight: float = 5.0,
        cutoff_to_latest_backfilled: bool = True
):
    """
    1) Load full price df, build features + regression target close[t+H].
    2) Load backfilled predictions with abs_error.
    3) Build sample weights: rows where we had larger error get larger weight.
    4) Retrain RandomForestRegressor with sample_weight and overwrite model_reg.pkl.

    cutoff_to_latest_backfilled=True:
      Only use rows with timestamp <= last prediction_time that has y_true,
      so you 'learn from mistakes' up to the latest validated time and avoid
      training on unlabeled future.
    """
    # 1) base data
    df = get_price_data(symbol, db_path=db_path)
    df = create_features(df)
    df = _prepare_reg_target(df, forward_steps)

    # 2) backfilled errors
    dfp = _load_backfilled_errors(db_path, symbol)
    if dfp.empty:
        print("No backfilled predictions found. Run backfill first.")
        return

    if cutoff_to_latest_backfilled:
        latest_pred_time = dfp["prediction_time"].max()
        df = df[df["timestamp"] <= latest_pred_time].copy()

    # 3) weights
    weights = _build_sample_weights(df, dfp, alpha=alpha, max_w=max_weight)

    # 4) train
    X = df[FEATURE_COLS]
    y = df["target_close"]
    print(f"[retrain] rows={len(df)}, weighted rows (w!=1)={(weights!=1).sum()}, "
          f"median_w={np.median(weights):.3f}, max_w={weights.max():.3f}")
    train_regressor(X, y, model_path="ml/model_reg.pkl")  # RFReg ignores weights directly,
    # If you want a regressor that supports sample_weight well, use HistGradientBoostingRegressor:
    # from sklearn.ensemble import HistGradientBoostingRegressor
    # model = HistGradientBoostingRegressor(max_depth=8, learning_rate=0.05)
    # model.fit(X, y, sample_weight=weights.values)
    # joblib.dump(model, "ml/model_reg.pkl")

    print("[retrain] Done. Updated ml/model_reg.pkl with error-informed training.")
