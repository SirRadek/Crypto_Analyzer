from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

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
from .xgb_price import build_bound, build_reg, clip_inside, to_price

structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso", key="ts"),
        structlog.processors.add_log_level,
        structlog.processors.JSONRenderer(),
    ]
)
logger = structlog.get_logger()


def _resolve_feature_names(cfg_or_list: Any) -> list[str]:
    """
    Accept either:
      - TrainConfig with features.path (JSON list of columns), or
      - sequence of feature names (list/tuple/pd.Index).
    """
    if isinstance(cfg_or_list, (list, tuple)):
        return [str(c) for c in cfg_or_list]
    try:
        import pandas as _pd  # lazy

        if isinstance(cfg_or_list, _pd.Index):
            return [str(c) for c in list(cfg_or_list)]
    except Exception:
        pass
    # Assume TrainConfig-like with .features.path
    path = Path(cfg_or_list.features.path)
    with open(path, encoding="utf-8") as f:
        names = json.load(f)
    if not isinstance(names, list):
        raise ValueError("feature list JSON must be a list[str]")
    return [str(x) for x in names]


def _to_f32(arr: Any) -> np.ndarray:
    """Return a float32 numpy array, densifying sparse inputs if needed."""
    return np.asarray(arr.toarray() if sparse.issparse(arr) else arr, dtype=np.float32)


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
    if X.dtypes.astype(str).ne("float32").any():
        raise ValueError("Features must be float32")
    if X.isna().any().any():
        raise ValueError("NaN values in features")
    return X


def train_price(
    df: pd.DataFrame, config: TrainConfig | list[str], outdir: str | Path
) -> tuple[dict[str, float], pd.DataFrame]:
    df = create_features(df)
    feature_cols = _resolve_feature_names(config)
    # When config is a list (tests), default to log target and sensible params.
    if isinstance(config, list):
        target_kind = "log"
        horizon_min = 120
        embargo = 24
        n_jobs = 1
    else:
        target_kind = config.target_kind
        horizon_min = config.horizon_min
        embargo = config.embargo
        n_jobs = config.n_jobs

    target_mid = "delta_log_120m" if target_kind == "log" else "delta_lin_120m"
    target_low = "delta_low_log_120m" if target_kind == "log" else "delta_low_lin_120m"
    target_high = "delta_high_log_120m" if target_kind == "log" else "delta_high_lin_120m"
    df = df.dropna(subset=[target_mid, target_low, target_high])
    X = _validate_features(df, feature_cols)
    y_mid = df[target_mid].astype("float32")
    y_low = df[target_low].astype("float32")
    y_high = df[target_high].astype("float32")

    zero_ratio = float((X == 0).sum().sum() / X.size)
    X_matrix = sparse.csr_matrix(X.values) if zero_ratio > 0.6 else X.values

    preds = []
    metrics = []
    reg_models, low_models, high_models = [], [], []
    start_time = time.monotonic()
    steps = horizon_min // 5

    for fold, (train_idx, test_idx) in enumerate(time_folds(len(df), embargo=embargo)):
        X_train, X_test = X_matrix[train_idx], X_matrix[test_idx]
        y_mid_train, y_mid_test = y_mid.iloc[train_idx], y_mid.iloc[test_idx]
        y_low_train, y_low_test = y_low.iloc[train_idx], y_low.iloc[test_idx]
        y_high_train, y_high_test = y_high.iloc[train_idx], y_high.iloc[test_idx]

        dtrain_mid = xgb.DMatrix(_to_f32(X_train), label=_to_f32(y_mid_train))
        dtest_mid = xgb.DMatrix(_to_f32(X_test), label=_to_f32(y_mid_test))
        dtrain_lo = xgb.DMatrix(_to_f32(X_train), label=_to_f32(y_low_train))
        dtest_lo = xgb.DMatrix(_to_f32(X_test), label=_to_f32(y_low_test))
        dtrain_hi = xgb.DMatrix(_to_f32(X_train), label=_to_f32(y_high_train))
        dtest_hi = xgb.DMatrix(_to_f32(X_test), label=_to_f32(y_high_test))

        reg_params, reg_rounds = build_reg()
        b_params, b_rounds = build_bound()
        for p in (reg_params, b_params):
            p["nthread"] = n_jobs

        reg = xgb.train(
            reg_params,
            dtrain_mid,
            reg_rounds,
            evals=[(dtest_mid, "test")],
            early_stopping_rounds=50,
            verbose_eval=False,
        )
        lowm = xgb.train(
            b_params,
            dtrain_lo,
            b_rounds,
            evals=[(dtest_lo, "test")],
            early_stopping_rounds=50,
            verbose_eval=False,
        )
        highm = xgb.train(
            b_params,
            dtrain_hi,
            b_rounds,
            evals=[(dtest_hi, "test")],
            early_stopping_rounds=50,
            verbose_eval=False,
        )
        reg_models.append(reg)
        low_models.append(lowm)
        high_models.append(highm)
        last_price = np.asarray(df["close"].iloc[test_idx].values, dtype=np.float32)
        delta_hat = reg.predict(dtest_mid)
        low_hat = lowm.predict(dtest_lo)
        high_hat = highm.predict(dtest_hi)
        p_hat = to_price(last_price, delta_hat, kind=target_kind)
        p_low = to_price(last_price, low_hat, kind=target_kind)
        p_high = to_price(last_price, high_hat, kind=target_kind)
        p_low, p_high = np.minimum(p_low, p_high), np.maximum(p_low, p_high)
        p_hat = clip_inside(p_hat, p_low, p_high)
        target_price = df["close"].shift(-steps).iloc[test_idx].values
        real_low = to_price(last_price, y_low_test.values, kind=target_kind)
        real_high = to_price(last_price, y_high_test.values, kind=target_kind)
        fold_df = pd.DataFrame(
            {
                "timestamp": df["timestamp"].iloc[test_idx].values,
                "p_low": p_low,
                "p_hat": p_hat,
                "p_high": p_high,
                "target": target_price,
                "real_low": real_low,
                "real_high": real_high,
                "last_price": last_price,
                "fold": fold,
            }
        )
        preds.append(fold_df)
        rmse = float(np.sqrt(np.mean((p_hat - target_price) ** 2)))
        mid_mae = float(np.mean(np.abs(p_hat - target_price)))
        coverage = float(np.mean((target_price >= p_low) & (target_price <= p_high)))
        width = float(np.mean(p_high - p_low))
        low_mae = float(np.mean(np.abs(p_low - real_low)))
        high_mae = float(np.mean(np.abs(p_high - real_high)))
        metrics.append(
            {
                "rmse": rmse,
                "mid_mae": mid_mae,
                "coverage": coverage,
                "width": width,
                "low_mae": low_mae,
                "high_mae": high_mae,
            }
        )
        _log(
            "fold",
            fold=fold,
            rmse=rmse,
            mid_mae=mid_mae,
            coverage=coverage,
            low_mae=low_mae,
            high_mae=high_mae,
        )

    avg_metrics = {
        "rmse": float(np.mean([m["rmse"] for m in metrics])),
        "mid_mae": float(np.mean([m["mid_mae"] for m in metrics])),
        "coverage": float(np.mean([m["coverage"] for m in metrics])),
        "width": float(np.mean([m["width"] for m in metrics])),
        "low_mae": float(np.mean([m["low_mae"] for m in metrics])),
        "high_mae": float(np.mean([m["high_mae"] for m in metrics])),
    }

    out_path = Path(outdir)
    out_path.mkdir(parents=True, exist_ok=True)

    reg_params, reg_rounds = build_reg()
    b_params, b_rounds = build_bound()
    for p in (reg_params, b_params):
        p["nthread"] = n_jobs
    dall_mid = xgb.DMatrix(_to_f32(X_matrix), label=_to_f32(y_mid))
    dall_lo = xgb.DMatrix(_to_f32(X_matrix), label=_to_f32(y_low))
    dall_hi = xgb.DMatrix(_to_f32(X_matrix), label=_to_f32(y_high))
    reg_final = xgb.train(reg_params, dall_mid, reg_rounds, verbose_eval=False)
    low_final = xgb.train(b_params, dall_lo, b_rounds, verbose_eval=False)
    high_final = xgb.train(b_params, dall_hi, b_rounds, verbose_eval=False)
    joblib.dump(reg_final, out_path / "reg.joblib")
    joblib.dump(low_final, out_path / "low.joblib")
    joblib.dump(high_final, out_path / "high.joblib")
    joblib.dump(
        {"reg": reg_models, "low": low_models, "high": high_models}, out_path / "ensemble.joblib"
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
        "horizon_min": horizon_min,
        "target_kind": target_kind,
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
        mid_mae=avg_metrics["mid_mae"],
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
