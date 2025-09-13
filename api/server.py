import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from fastapi import FastAPI

from analysis.feature_engineering import FEATURE_COLUMNS, create_features
from crypto_analyzer.schemas import PredictionResponse
from ml.xgb_price import clip_inside, to_price

MODEL_DIR = Path(os.getenv("MODEL_DIR", "models/xgb_price"))
DATA_PATH = os.getenv("DATA_PATH", "data/latest.parquet")

app = FastAPI()

reg = joblib.load(MODEL_DIR / "reg.joblib", mmap_mode="r")
lo = joblib.load(MODEL_DIR / "low.joblib", mmap_mode="r")
hi = joblib.load(MODEL_DIR / "high.joblib", mmap_mode="r")


def _load_last_row() -> tuple[pd.Timestamp, float, pd.DataFrame]:
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


@app.get("/predict", response_model=PredictionResponse)  # type: ignore[misc]
def predict() -> PredictionResponse:  # pragma: no cover - FastAPI handles response
    ts, last_price, X_last = _load_last_row()
    dlast = xgb.DMatrix(np.asarray(X_last, dtype=np.float32))
    delta = float(reg.predict(dlast)[0])
    low = float(lo.predict(dlast)[0])
    high = float(hi.predict(dlast)[0])
    p_hat = float(to_price(last_price, delta))
    p_low = float(to_price(last_price, low))
    p_high = float(to_price(last_price, high))
    p_low, p_high = (min(p_low, p_high), max(p_low, p_high))
    p_hat = clip_inside(
        np.array([p_hat], dtype=np.float32),
        np.array([p_low], dtype=np.float32),
        np.array([p_high], dtype=np.float32),
    )[0]
    return PredictionResponse(
        timestamp=str(ts),
        p_low=float(p_low),
        p_hat=float(p_hat),
        p_high=float(p_high),
    )
