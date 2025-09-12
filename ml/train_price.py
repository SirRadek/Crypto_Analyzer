from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import psutil
import structlog
import xgboost as xgb
import yaml
from scipy import sparse

from analysis.feature_engineering import create_features
from crypto_analyzer.schemas import TrainConfig

from .backtest import run_backtest
from .time_cv import time_folds
from .xgb_price import build_quantile, build_reg, clip_inside, to_price

structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso", key="ts"),
        structlog.processors.add_log_level,
        structlog.processors.JSONRenderer(),
    ]
)
logger = structlog.get_logger()


def _log(stage: str, **kwargs: float | int) -> None:
    process = psutil.Process()
    payload = {
        "stage": stage,
        "peak_ram_mb": process.memory_info().rss / 1e6,
    }
    payload.update(kwargs)
    logger.info(stage, **payload)


def load_config(path: Path) -> TrainConfig:
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return TrainConfig(**data)


def _validate_features(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    missing = set(feature_cols) - set(df.columns)
    if missing:
        raise ValueError(f"Missing features: {missing}")
    X = df[feature_cols]
    if X.dtypes.ne(np.float32).any():
        raise ValueError("Features must be float32")
    if X.isna().any().any():
        raise ValueError("NaN values in features")
    return X


def train_price(
    df: pd.DataFrame, config: TrainConfig, outdir: str | Path
) -> tuple[dict[str, float], pd.DataFrame]:
    df = create_features(df)
    with open(config.features.path, encoding="utf-8") as f:
        feature_cols = json.load(f)
    target_col = "delta_log_120m" if config.target_kind == "log" else "delta_lin_120m"
    df = df.dropna(subset=[target_col])
    X = _validate_features(df, feature_cols)
    y = df[target_col].astype("float32")

    zero_ratio = float((X == 0).sum().sum() / X.size)
    X_matrix = sparse.csr_matrix(X.values) if zero_ratio > 0.6 else X.values

    preds = []
    metrics = []
    reg_models, q10_models, q90_models = [], [], []
    start_time = time.monotonic()
    steps = config.horizon_min // 5

    for fold, (train_idx, test_idx) in enumerate(time_folds(len(df), embargo=config.embargo)):
        X_train, y_train = X_matrix[train_idx], y.iloc[train_idx]
        X_test, y_test = X_matrix[test_idx], y.iloc[test_idx]
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dtest = xgb.DMatrix(X_test, label=y_test)
        reg_params, reg_rounds = build_reg()
        q10_params, q_rounds = build_quantile(config.quantiles["low"])
        q90_params, q90_rounds = build_quantile(config.quantiles["high"])
        reg_params["nthread"] = config.n_jobs
        q10_params["nthread"] = config.n_jobs
        q90_params["nthread"] = config.n_jobs
        reg = xgb.train(
            reg_params,
            dtrain,
            reg_rounds,
            evals=[(dtest, "test")],
            early_stopping_rounds=50,
            verbose_eval=False,
        )
        q10 = xgb.train(
            q10_params,
            dtrain,
            q_rounds,
            evals=[(dtest, "test")],
            early_stopping_rounds=50,
            verbose_eval=False,
        )
        q90 = xgb.train(
            q90_params,
            dtrain,
            q90_rounds,
            evals=[(dtest, "test")],
            early_stopping_rounds=50,
            verbose_eval=False,
        )
        reg_models.append(reg)
        q10_models.append(q10)
        q90_models.append(q90)
        last_price = df["close"].iloc[test_idx].values
        delta_hat = reg.predict(dtest)
        low_hat = q10.predict(dtest)
        high_hat = q90.predict(dtest)
        p_hat = to_price(last_price, delta_hat, kind=config.target_kind)
        p_low = to_price(last_price, low_hat, kind=config.target_kind)
        p_high = to_price(last_price, high_hat, kind=config.target_kind)
        p_low, p_high = np.minimum(p_low, p_high), np.maximum(p_low, p_high)
        p_hat = clip_inside(p_hat, p_low, p_high)
        target_price = df["close"].shift(-steps).iloc[test_idx].values
        fold_df = pd.DataFrame(
            {
                "timestamp": df["timestamp"].iloc[test_idx].values,
                "p_low": p_low,
                "p_hat": p_hat,
                "p_high": p_high,
                "target": target_price,
                "last_price": last_price,
                "fold": fold,
            }
        )
        preds.append(fold_df)
        rmse = float(np.sqrt(np.mean((p_hat - target_price) ** 2)))
        mae = float(np.mean(np.abs(p_hat - target_price)))
        coverage = float(np.mean((target_price >= p_low) & (target_price <= p_high)))
        width = float(np.mean(p_high - p_low))
        metrics.append({"rmse": rmse, "mae": mae, "coverage": coverage, "width": width})
        _log("fold", fold=fold, rmse=rmse, mae=mae, coverage=coverage)

    avg_metrics = {
        "rmse": float(np.mean([m["rmse"] for m in metrics])),
        "mae": float(np.mean([m["mae"] for m in metrics])),
        "coverage": float(np.mean([m["coverage"] for m in metrics])),
        "width": float(np.mean([m["width"] for m in metrics])),
    }

    out_path = Path(outdir)
    out_path.mkdir(parents=True, exist_ok=True)

    reg_params, reg_rounds = build_reg()
    q10_params, q_rounds = build_quantile(config.quantiles["low"])
    q90_params, q90_rounds = build_quantile(config.quantiles["high"])
    for p in (reg_params, q10_params, q90_params):
        p["nthread"] = config.n_jobs
    dall = xgb.DMatrix(X_matrix, label=y)
    reg_final = xgb.train(reg_params, dall, reg_rounds, verbose_eval=False)
    q10_final = xgb.train(q10_params, dall, q_rounds, verbose_eval=False)
    q90_final = xgb.train(q90_params, dall, q90_rounds, verbose_eval=False)
    joblib.dump(reg_final, out_path / "reg.joblib")
    joblib.dump(q10_final, out_path / "q10.joblib")
    joblib.dump(q90_final, out_path / "q90.joblib")
    joblib.dump(
        {"reg": reg_models, "q10": q10_models, "q90": q90_models}, out_path / "ensemble.joblib"
    )

    pred_df = pd.concat(preds, ignore_index=True).sort_values("timestamp")
    pred_df.to_csv(out_path / "cv_preds.csv", index=False)
    bt = run_backtest(pred_df)
    bt["equity"].to_csv(out_path / "backtest.csv", index=False)
    _log("backtest", pnl=bt["metrics"]["pnl"], sharpe=bt["metrics"]["sharpe"])

    importance = reg_final.get_score(importance_type="gain")
    fi = pd.DataFrame({"feature": list(importance.keys()), "importance": list(importance.values())})
    fi.to_csv(out_path / "feature_importance.csv", index=False)

    meta = {
        "horizon_min": config.horizon_min,
        "target_kind": config.target_kind,
        "quantiles": config.quantiles,
        "data": {"n_samples": int(len(df))},
        "metrics": avg_metrics,
        "backtest": bt["metrics"],
    }
    with open(out_path / "model_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f)
    with open(out_path / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(avg_metrics, f)

    elapsed = time.monotonic() - start_time
    _log(
        "train_complete",
        n_samples=len(df),
        rmse=avg_metrics["rmse"],
        mae=avg_metrics["mae"],
        coverage=avg_metrics["coverage"],
        pnl=bt["metrics"]["pnl"],
        sharpe=bt["metrics"]["sharpe"],
        runtime_sec=elapsed,
    )

    return avg_metrics, pred_df


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train XGBoost price models")
    parser.add_argument("--config", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--outdir", default="models/xgb_price")
    args = parser.parse_args(argv)

    config = load_config(Path(args.config))
    df = pd.read_parquet(args.data) if args.data.endswith(".parquet") else pd.read_csv(args.data)
    train_price(df, config, args.outdir)


if __name__ == "__main__":
    main()
