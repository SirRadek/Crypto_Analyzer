import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from scipy import sparse

from analysis.feature_engineering import create_features

from .backtest import run_backtest
from .time_cv import time_folds
from .xgb_price import build_quantile, build_reg, clip_inside, to_price


def train_price(df, feature_cols, target_kind="log", outdir="models/xgb_price"):
    df = create_features(df)
    target_col = "delta_log_120m" if target_kind == "log" else "delta_lin_120m"
    df = df.dropna(subset=[target_col])
    X = df[feature_cols].astype("float32")
    y = df[target_col].astype("float32")

    zero_ratio = float((X == 0).sum().sum() / X.size)
    X_matrix = sparse.csr_matrix(X.values) if zero_ratio > 0.5 else X.values

    preds = []
    metrics = []
    reg_models, q10_models, q90_models = [], [], []

    for fold, (train_idx, test_idx) in enumerate(time_folds(len(df), embargo=24)):
        X_train, y_train = X_matrix[train_idx], y.iloc[train_idx]
        X_test, y_test = X_matrix[test_idx], y.iloc[test_idx]
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dtest = xgb.DMatrix(X_test, label=y_test)
        reg_params, reg_rounds = build_reg()
        q10_params, q_rounds = build_quantile(0.10)
        q90_params, q90_rounds = build_quantile(0.90)
        reg = xgb.train(reg_params, dtrain, reg_rounds, evals=[(dtest, "test")], verbose_eval=False)
        q10 = xgb.train(q10_params, dtrain, q_rounds, evals=[(dtest, "test")], verbose_eval=False)
        q90 = xgb.train(q90_params, dtrain, q90_rounds, evals=[(dtest, "test")], verbose_eval=False)
        reg_models.append(reg)
        q10_models.append(q10)
        q90_models.append(q90)
        last_price = df["close"].iloc[test_idx].values
        delta_hat = reg.predict(dtest)
        low_hat = q10.predict(dtest)
        high_hat = q90.predict(dtest)
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
                    "last_price": last_price,
                    "fold": fold,
                }
            )
        )
        rmse = float(np.sqrt(np.mean((p_hat - target_price) ** 2)))
        mae = float(np.mean(np.abs(p_hat - target_price)))
        coverage = float(np.mean((target_price >= p_low) & (target_price <= p_high)))
        width = float(np.mean(p_high - p_low))
        metrics.append({"rmse": rmse, "mae": mae, "coverage": coverage, "width": width})

    avg_metrics = {
        "rmse": float(np.mean([m["rmse"] for m in metrics])),
        "mae": float(np.mean([m["mae"] for m in metrics])),
        "coverage": float(np.mean([m["coverage"] for m in metrics])),
        "width": float(np.mean([m["width"] for m in metrics])),
    }

    out_path = Path(outdir)
    out_path.mkdir(parents=True, exist_ok=True)

    reg_params, reg_rounds = build_reg()
    q10_params, q_rounds = build_quantile(0.10)
    q90_params, q90_rounds = build_quantile(0.90)
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

    importance = reg_final.get_score(importance_type="gain")
    fi = pd.DataFrame({"feature": list(importance.keys()), "importance": list(importance.values())})
    fi.to_csv(out_path / "feature_importance.csv", index=False)

    try:
        import importlib
        import os

        if os.getenv("ENABLE_SHAP") and importlib.util.find_spec("shap"):
            import matplotlib.pyplot as plt
            import shap  # type: ignore

            sample = X.tail(1000)
            explainer = shap.TreeExplainer(reg_final)
            shap_values = explainer.shap_values(sample)
            shap.summary_plot(shap_values, sample, show=False)
            plt.savefig(out_path / "shap_summary.png", bbox_inches="tight")
            plt.close()
    except Exception:
        pass

    meta = {
        "n_samples": int(len(df)),
        "horizon": 120,
        "metrics": avg_metrics,
        "train_start": str(df["timestamp"].min()),
        "train_end": str(df["timestamp"].max()),
        "libraries": {
            "xgboost": xgb.__version__,
            "pandas": pd.__version__,
            "numpy": np.__version__,
        },
        "backtest": bt["metrics"],
    }
    with open(out_path / "model_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f)

    with open(out_path / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(avg_metrics, f)

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
