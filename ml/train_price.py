import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from analysis.feature_engineering import create_features

from .time_cv import time_folds
from .xgb_price import build_quantile, build_reg, clip_inside, to_price


def train_price(df, feature_cols, target_kind="log", outdir="models/xgb_price"):
    df = create_features(df)
    target_col = "delta_log_120m" if target_kind == "log" else "delta_lin_120m"
    df = df.dropna(subset=[target_col])
    X = df[feature_cols].astype("float32")
    y = df[target_col].astype("float32")

    preds = []
    metrics = []

    for train_idx, test_idx in time_folds(len(df), embargo=24):
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]
        reg = build_reg()
        q10 = build_quantile(0.10)
        q90 = build_quantile(0.90)
        reg.fit(
            X_train,
            y_train,
            eval_set=[(X_test, y_test)],
            verbose=False,
        )
        q10.fit(
            X_train,
            y_train,
            eval_set=[(X_test, y_test)],
            verbose=False,
        )
        q90.fit(
            X_train,
            y_train,
            eval_set=[(X_test, y_test)],
            verbose=False,
        )
        last_price = df["close"].iloc[test_idx].values
        delta_hat = reg.predict(X_test)
        low_hat = q10.predict(X_test)
        high_hat = q90.predict(X_test)
        p_hat = to_price(last_price, delta_hat, kind=target_kind)
        p_low = to_price(last_price, low_hat, kind=target_kind)
        p_high = to_price(last_price, high_hat, kind=target_kind)
        p_low, p_high = np.minimum(p_low, p_high), np.maximum(p_low, p_high)
        p_hat = clip_inside(p_hat, p_low, p_high)
        target_price = df["close"].shift(-24).iloc[test_idx].values
        preds.append(
            pd.DataFrame(
                {
                    "timestamp": df["timestamp"].iloc[test_idx].values,
                    "p_low": p_low,
                    "p_hat": p_hat,
                    "p_high": p_high,
                    "target": target_price,
                }
            )
        )
        rmse = float(np.sqrt(np.mean((p_hat - target_price) ** 2)))
        mae = float(np.mean(np.abs(p_hat - target_price)))
        coverage = float(np.mean((target_price >= p_low) & (target_price <= p_high)))
        width = float(np.mean(p_high - p_low))
        metrics.append(dict(rmse=rmse, mae=mae, coverage=coverage, width=width))

    avg_metrics = {
        "rmse": float(np.mean([m["rmse"] for m in metrics])),
        "mae": float(np.mean([m["mae"] for m in metrics])),
        "coverage": float(np.mean([m["coverage"] for m in metrics])),
        "width": float(np.mean([m["width"] for m in metrics])),
    }

    out_path = Path(outdir)
    out_path.mkdir(parents=True, exist_ok=True)

    reg_final = build_reg()
    q10_final = build_quantile(0.10)
    q90_final = build_quantile(0.90)
    reg_final.fit(X, y, verbose=False)
    q10_final.fit(X, y, verbose=False)
    q90_final.fit(X, y, verbose=False)
    joblib.dump(reg_final, out_path / "reg.joblib")
    joblib.dump(q10_final, out_path / "q10.joblib")
    joblib.dump(q90_final, out_path / "q90.joblib")
    with open(out_path / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(avg_metrics, f)

    pred_df = pd.concat(preds, ignore_index=True).sort_values("timestamp")
    return avg_metrics, pred_df


def main(argv=None):
    parser = argparse.ArgumentParser(description="Train XGBoost price models")
    parser.add_argument("--horizon-min", type=int, default=120)
    parser.add_argument("--features", required=True)
    parser.add_argument("--target-kind", choices=["log", "lin"], default="log")
    parser.add_argument("--outdir", default="models/xgb_price")
    parser.add_argument("--data")
    args = parser.parse_args(argv)

    if args.data is None:
        raise SystemExit("--data path required")
    df = pd.read_parquet(args.data) if args.data.endswith(".parquet") else pd.read_csv(args.data)
    with open(args.features, encoding="utf-8") as f:
        feature_cols = json.load(f)
    metrics, _ = train_price(df, feature_cols, target_kind=args.target_kind, outdir=args.outdir)
    print(json.dumps(metrics))


if __name__ == "__main__":
    main()
