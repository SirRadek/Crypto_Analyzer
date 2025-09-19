from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from analysis.feature_engineering import assign_feature_groups, create_features
from db.db_connector import get_price_data
from utils.config import CONFIG, override_feature_settings
from utils.helpers import ensure_dir_exists, get_logger, set_cpu_limit
from utils.progress import timed
from utils.timeframes import interval_to_minutes

logger = get_logger(__name__)

SYMBOL = CONFIG.symbol
DB_PATH = CONFIG.db_path
INTERVAL = CONFIG.interval
INTERVAL_MINUTES = interval_to_minutes(INTERVAL)
CPU_LIMIT = CONFIG.cpu_limit


def prepare_targets(df: pd.DataFrame, forward_steps: int = 1) -> pd.DataFrame:
    """Create binary classification targets ``forward_steps`` ahead."""

    df = df.copy(deep=False)
    df["target_cls"] = (df["close"].shift(-forward_steps) > df["close"]).astype("int8")
    df = df.dropna(subset=["target_cls"])
    return df


def _build_feature_matrix(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    base_cols = {
        "timestamp",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "quote_asset_volume",
        "taker_buy_base",
        "taker_buy_quote",
        "number_of_trades",
    }
    feature_cols = [c for c in df.columns if c not in base_cols.union({"target"})]
    X = df[feature_cols].astype(np.float32)
    y = df["target"].astype(np.int8)
    return X, y


def run_pipeline(
    task: str,
    horizon: int,
    use_onchain: bool | None,
    txn_cost_bps: float,
    split_params: dict[str, Any],
    gpu: bool,
    out_dir: str,
) -> None:
    if task != "clf":
        raise ValueError("Only classification pipeline is supported")

    ensure_dir_exists(out_dir)

    feature_settings = CONFIG.features
    if use_onchain is not None:
        feature_settings = override_feature_settings(
            feature_settings, include_onchain=use_onchain
        )

    df = get_price_data(SYMBOL, db_path=DB_PATH)
    df = create_features(df, settings=feature_settings)

    if horizon % INTERVAL_MINUTES != 0:
        raise ValueError(
            "Prediction horizon must be divisible by the candle interval: "
            f"{horizon}m vs {INTERVAL}"
        )
    horizon_steps = horizon // INTERVAL_MINUTES
    df["target"] = (df["close"].shift(-horizon_steps) > df["close"]).astype(np.int8)
    df = df.dropna(subset=["target"])

    X, y = _build_feature_matrix(df)

    split_cfg = {"test_size": 0.2, "shuffle": False}
    split_cfg.update(split_params)
    X_train, X_test, y_train, y_test = train_test_split(X, y, **split_cfg)

    params = {
        "n_estimators": 200,
        "tree_method": "gpu_hist" if gpu else "hist",
        "eval_metric": "logloss",
        "random_state": 42,
    }
    model = xgb.XGBClassifier(**params)
    model.fit(X_train, y_train)

    probas = model.predict_proba(X_test)[:, 1]
    preds = (probas >= 0.5).astype(np.int8)
    accuracy = float(accuracy_score(y_test, preds))

    out_path = Path(out_dir)
    metrics = {"accuracy": accuracy}
    with (out_path / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    pred_df = pd.DataFrame(
        {
            "timestamp": df.loc[y_test.index, "timestamp"].reset_index(drop=True),
            "y_true": y_test.reset_index(drop=True),
            "y_pred_proba": probas,
            "y_pred": preds,
        }
    )
    pred_df.to_csv(out_path / "predictions_clf.csv", index=False)

    try:  # optional explainability artefacts
        import matplotlib.pyplot as plt
        import shap
        from sklearn.inspection import permutation_importance

        groups = assign_feature_groups(list(X.columns))

        perm = permutation_importance(
            model, X_test, y_test, n_repeats=5, random_state=42
        )
        perm_df = pd.DataFrame(
            {"feature": X.columns, "importance": perm.importances_mean}
        ).sort_values("importance", ascending=False)
        perm_df.to_csv(out_path / "perm_importance_clf.csv", index=False)

        explainer = shap.Explainer(model, X_train)
        shap_vals = explainer(X_test).values
        shap_mean = (
            np.abs(shap_vals)
            .mean(axis=0)
        )
        shap_df = pd.DataFrame(
            {"feature": X.columns, "mean_abs_shap": shap_mean}
        ).sort_values("mean_abs_shap", ascending=False)
        shap_df.to_csv(out_path / "shap_values_clf.csv", index=False)

        shap_df["group"] = shap_df["feature"].map(groups)
        shap_group = shap_df.groupby("group")["mean_abs_shap"].sum().sort_values(ascending=False)
        shap_group.to_csv(out_path / "shap_group_clf.csv")

        perm_df["group"] = perm_df["feature"].map(groups)
        perm_group = perm_df.groupby("group")["importance"].sum().sort_values(ascending=False)
        perm_group.to_csv(out_path / "perm_group_clf.csv")

        fig, ax = plt.subplots()
        shap_df.head(20).plot.barh(x="feature", y="mean_abs_shap", ax=ax)
        fig.tight_layout()
        fig.savefig(out_path / "shap_top20_clf.png")
        plt.close(fig)

        fig, ax = plt.subplots()
        perm_df.head(20).plot.barh(x="feature", y="importance", ax=ax)
        fig.tight_layout()
        fig.savefig(out_path / "perm_top20_clf.png")
        plt.close(fig)

    except Exception:  # pragma: no cover - optional deps may be missing
        logger.info("Skipping explainability outputs due to missing dependencies")

    booster_path = out_path / "clf_model.json"
    model.get_booster().save_model(booster_path)

    run_cfg = {
        "task": task,
        "horizon": horizon,
        "use_onchain": feature_settings.include_onchain,
        "txn_cost_bps": txn_cost_bps,
        "split_params": split_params,
        "gpu": gpu,
        "out_dir": out_dir,
        "random_state": 42,
    }
    with (out_path / "run_config.json").open("w", encoding="utf-8") as f:
        json.dump(run_cfg, f, indent=2)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the classification pipeline")
    parser.add_argument("--task", default="clf")
    parser.add_argument("--horizon", type=int, default=120)
    parser.add_argument("--use_onchain", action="store_true")
    parser.add_argument("--txn_cost_bps", type=float, default=1.0)
    parser.add_argument("--split_params", type=str, default="{}")
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--out_dir", type=str, default="outputs")
    parser.add_argument("--cpu_limit", type=int, default=CPU_LIMIT)
    return parser.parse_args(argv)


def _main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.cpu_limit:
        set_cpu_limit(args.cpu_limit)

    try:
        split_params = json.loads(args.split_params)
    except json.JSONDecodeError as exc:  # pragma: no cover - defensive
        raise SystemExit(f"Invalid --split_params JSON: {exc}") from exc

    with timed("train_pipeline"):
        run_pipeline(
            task=args.task,
            horizon=args.horizon,
            use_onchain=True if args.use_onchain else None,
            txn_cost_bps=args.txn_cost_bps,
            split_params=split_params,
            gpu=args.gpu,
            out_dir=args.out_dir,
        )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_main())
