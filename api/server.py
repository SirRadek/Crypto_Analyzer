import json
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from fastapi import FastAPI

from analysis.feature_engineering import FEATURE_COLUMNS, create_features
from crypto_analyzer.schemas import PredictionResponse
from ml.xgb_price import to_price

MODEL_DIR = Path(os.getenv("MODEL_DIR", "models/xgb_price"))
DATA_PATH = os.getenv("DATA_PATH", "data/latest.parquet")
FEATURE_LIST_PATH = Path(os.getenv("FEATURE_LIST_PATH", "analysis/feature_list.json"))

app = FastAPI()

reg = joblib.load(MODEL_DIR / "reg.joblib", mmap_mode="r")


def _feature_names() -> list[str]:
    """Return the feature order used during model training."""

    if FEATURE_LIST_PATH.exists():
        try:
            with open(FEATURE_LIST_PATH, encoding="utf-8") as f:
                names = json.load(f)
            if isinstance(names, list):
                return [str(n) for n in names]
        except (OSError, json.JSONDecodeError):
            pass
    return FEATURE_COLUMNS


def _load_last_row() -> tuple[pd.Timestamp, float, pd.DataFrame]:
    df = (
        pd.read_parquet(DATA_PATH)
        if DATA_PATH.endswith(".parquet")
        else pd.read_csv(DATA_PATH, parse_dates=["timestamp"])
    )
    df = create_features(df)
    row = df.iloc[[-1]]
    last_price = float(row["close"].iloc[0])
    feature_cols = _feature_names()
    X_last = row[feature_cols].astype("float32")
    ts = row["timestamp"].iloc[0]
    return ts, last_price, X_last


@app.get("/predict", response_model=PredictionResponse)  # type: ignore[misc]
def predict() -> PredictionResponse:  # pragma: no cover - FastAPI handles response
    ts, last_price, X_last = _load_last_row()
    dlast = xgb.DMatrix(np.asarray(X_last, dtype=np.float32))
    delta = float(reg.predict(dlast)[0])
    p_hat = float(to_price(last_price, delta))
    return PredictionResponse(timestamp=str(ts), p_hat=p_hat)
