#!/usr/bin/env python
"""Train + backtest across horizons/seeds and save a summary CSV."""

from __future__ import annotations

import subprocess
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from crypto_analyzer.eval.backtest import run_backtest
from crypto_analyzer.models.utils import match_model_features


HORIZONS = (15, 30, 60, 120, 240)
SEEDS = (41, 42, 43, 44, 45)
EARLY_STOPPING_ROUNDS = 50

FEATURES_PATH = Path("data/features.parquet")
MODEL_PATH = Path("artifacts/meta_model.joblib")
OUTPUT_PATH = Path("reports/horizon_sweep.csv")


def _predict_proba(model: object, X: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "get_booster"):
        try:
            booster = model.get_booster()
            try:
                proba = booster.inplace_predict(X, predict_type="probability", device="cuda")
            except TypeError:
                proba = booster.inplace_predict(X, device="cuda")
            arr = np.asarray(proba)
            if arr.ndim == 1:
                arr = np.column_stack([1.0 - arr, arr])
            return arr
        except Exception:
            pass

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        arr = np.asarray(proba)
        if arr.ndim == 1:
            arr = np.column_stack([1.0 - arr, arr])
        return arr

    preds = np.asarray(model.predict(X)).ravel()
    return np.column_stack([1.0 - preds, preds])


def _train_model(horizon: int, seed: int) -> None:
    subprocess.run(
        [
            "tools/run_gpu.sh",
            "scripts.train",
            "--features",
            str(FEATURES_PATH),
            "--model-path",
            str(MODEL_PATH),
            "--use-cache",
            "--early-stopping-rounds",
            str(EARLY_STOPPING_ROUNDS),
            "--horizon",
            str(horizon),
            "--random-state",
            str(seed),
        ],
        check=True,
    )


def _infer_step_minutes(timestamps: pd.Series) -> int:
    diffs = timestamps.diff().dropna().median()
    step_min = int(diffs.total_seconds() // 60) if isinstance(diffs, pd.Timedelta) else 1
    return max(step_min, 1)


def _build_predictions(
    features: pd.DataFrame, model: object, horizon: int
) -> pd.DataFrame:
    step_min = _infer_step_minutes(pd.to_datetime(features["timestamp"], utc=True, errors="coerce"))
    periods = max(1, int(round(horizon / step_min)))
    future_close = features["close"].shift(-periods)
    valid = future_close.notna()

    X = match_model_features(features.loc[valid].drop(columns=["timestamp"], errors="ignore"), model)
    proba = _predict_proba(model, X)
    p_hat = proba[:, 1] if proba.ndim == 2 else proba.ravel()

    return pd.DataFrame(
        {
            "timestamp": features.loc[valid, "timestamp"].values,
            "p_hat": p_hat,
            "target": future_close.loc[valid].astype(float).values,
            "last_price": features.loc[valid, "close"].astype(float).values,
        }
    )


def _backtest(predictions: pd.DataFrame) -> dict[str, float]:
    result = run_backtest(
        predictions,
        prob_col="p_hat",
        p_up_threshold=0.65,
        fee_bps=4.0,
        slippage_bps=0.0,
        position_size=0.10,
        max_leverage=2.0,
        max_trade_loss=0.06,
        max_trade_gain=0.30,
        compound=False,
    )
    return result["metrics"]


def main() -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text("horizon,seed,pnl,ev,sharpe,maxDD,hit_rate,trades\n")

    features = pd.read_parquet(FEATURES_PATH).sort_values("timestamp").reset_index(drop=True)
    total_runs = len(HORIZONS) * len(SEEDS)
    completed = 0

    for horizon in HORIZONS:
        for seed in SEEDS:
            print(f"START horizon={horizon} seed={seed}")
            _train_model(horizon, seed)
            model = joblib.load(MODEL_PATH)
            preds = _build_predictions(features, model, horizon)
            metrics = _backtest(preds)
            row = pd.DataFrame(
                [
                    {
                        "horizon": horizon,
                        "seed": seed,
                        "pnl": metrics["pnl"],
                        "ev": metrics["ev"],
                        "sharpe": metrics["sharpe"],
                        "maxDD": metrics["max_drawdown"],
                        "hit_rate": metrics["hit_rate"],
                        "trades": metrics["trades"],
                    }
                ]
            )
            row.to_csv(OUTPUT_PATH, mode="a", header=False, index=False)
            completed += 1
            print(f"DONE horizon={horizon} seed={seed} ({completed}/{total_runs})")


if __name__ == "__main__":
    main()
