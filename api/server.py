import os
from pathlib import Path

import joblib
import pandas as pd
import xgboost as xgb
from fastapi import FastAPI

from analysis.feature_engineering import FEATURE_COLUMNS, create_features
from ml.xgb_price import clip_inside, to_price

MODEL_DIR = Path(os.getenv("MODEL_DIR", "models/xgb_price"))
DATA_PATH = os.getenv("DATA_PATH", "data/latest.parquet")

app = FastAPI()

reg = joblib.load(MODEL_DIR / "reg.joblib", mmap_mode="r")
q10 = joblib.load(MODEL_DIR / "q10.joblib", mmap_mode="r")
q90 = joblib.load(MODEL_DIR / "q90.joblib", mmap_mode="r")


def _load_last_row():
    df = (
        pd.read_parquet(DATA_PATH)
        if DATA_PATH.endswith(".parquet")
        else pd.read_csv(DATA_PATH, parse_dates=["timestamp"])
    )
    df = create_features(df)
    row = df.iloc[[-1]]
    last_price = float(row["close"].iloc[0])
    X_last = row[FEATURE_COLUMNS].astype("float32")
    ts = row["timestamp"].iloc[0]
    return ts, last_price, X_last


@app.get("/predict")
def predict():
    ts, last_price, X_last = _load_last_row()
    dlast = xgb.DMatrix(X_last)
    delta = reg.predict(dlast)[0]
    low = q10.predict(dlast)[0]
    high = q90.predict(dlast)[0]
    p_hat = to_price(last_price, delta)
    p_low = to_price(last_price, low)
    p_high = to_price(last_price, high)
    p_low, p_high = min(p_low, p_high), max(p_low, p_high)
    p_hat = clip_inside(p_hat, p_low, p_high)
    return {
        "timestamp": str(ts),
        "p_low": float(p_low),
        "p_hat": float(p_hat),
        "p_high": float(p_high),
    }
