from __future__ import annotations

import argparse
import json
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
for candidate in (ROOT, SRC):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from crypto_analyzer.features.engineering import assign_feature_groups, create_features
from crypto_analyzer.labeling.targets import make_targets
from crypto_analyzer.data.db_connector import get_price_data
from crypto_analyzer.utils.config import CONFIG, config_to_dict, override_feature_settings
from crypto_analyzer.utils.helpers import ensure_dir_exists, get_logger, set_cpu_limit
from crypto_analyzer.utils.progress import timed
from crypto_analyzer.utils.timeframes import interval_to_minutes

logger = get_logger(__name__)

CPU_LIMIT = CONFIG.cpu_limit

def _build_feature_matrix(df: pd.DataFrame, target_col: str) -> tuple[pd.DataFrame, pd.Series]:
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
    target_columns = {target_col}
    target_columns.update(
        c for c in df.columns if c.startswith("cls_sign_") or c.startswith("beyond_costs_")
    )
    feature_cols = [c for c in df.columns if c not in base_cols.union(target_columns)]
    X = df[feature_cols].astype(np.float32)
    y = df[target_col].astype(np.int8)
    return X, y


def prepare_targets(
    df: pd.DataFrame,
    forward_steps: int = 1,
    *,
    txn_cost_bps: float = 1.0,
) -> pd.DataFrame:
    """Create a DataFrame with the binary target ``forward_steps`` ahead."""

    if "timestamp" not in df.columns:
        raise KeyError("Input frame must contain a 'timestamp' column")

    ts = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    if ts.isna().any():
        raise ValueError("Timestamp column contains non-parsable values")
    deltas = ts.diff().dropna()
    if len(deltas) == 0:
        step_minutes = interval_to_minutes(CONFIG.interval)
    else:
        median_delta = deltas.median()
        step_minutes = (
            int(round(median_delta.total_seconds() / 60)) if isinstance(median_delta, pd.Timedelta) else 1
        )
        step_minutes = max(step_minutes, 1)

    horizon_minutes = max(1, forward_steps) * step_minutes
    labeled = make_targets(df, horizons_min=[horizon_minutes], txn_cost_bps=txn_cost_bps)
    target_col = f"cls_sign_{horizon_minutes}m"
    if target_col not in labeled.columns:
        raise KeyError(f"Target column {target_col!r} missing from generated targets")

    if forward_steps > 0:
        if len(labeled) <= forward_steps:
            raise ValueError("Not enough rows to build forward-looking targets")
        labeled = labeled.iloc[:-forward_steps, :].copy()

    labeled = labeled.dropna(subset=[target_col]).reset_index(drop=True)
    drop_cols = [c for c in labeled.columns if c.startswith("beyond_costs_")]
    if drop_cols:
        labeled = labeled.drop(columns=drop_cols)
    labeled = labeled.rename(columns={target_col: "target_cls"})
    return labeled


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

    base_out_dir = ensure_dir_exists(Path(out_dir))
    run_id = f"{datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')}_{uuid4().hex[:8]}"
    run_dir = ensure_dir_exists(base_out_dir / f"run_id={run_id}")

    random_seed = CONFIG.models.random_seed
    random.seed(random_seed)
    np.random.seed(random_seed)

    feature_settings = CONFIG.features
    if use_onchain is not None:
        feature_settings = override_feature_settings(
            feature_settings, include_onchain=use_onchain
        )

    interval_minutes = interval_to_minutes(CONFIG.interval)
    if horizon % interval_minutes != 0:
        raise ValueError(
            "Prediction horizon must be divisible by the candle interval: "
            f"{horizon}m vs {CONFIG.interval}"
        )
    horizon_steps = max(1, horizon // interval_minutes)

    df = get_price_data(CONFIG.symbol, db_path=CONFIG.db_path)
    df = create_features(df, settings=feature_settings)
    df = make_targets(df, horizons_min=[horizon], txn_cost_bps=txn_cost_bps)
    target_col = f"cls_sign_{horizon}m"
    if target_col not in df.columns:
        raise KeyError(f"Target column {target_col!r} missing from target generation")
    if horizon_steps > 0:
        if len(df) <= horizon_steps:
            raise ValueError("Not enough samples for the requested prediction horizon")
        df = df.iloc[:-horizon_steps, :].copy()

    df = df.dropna(subset=[target_col]).reset_index(drop=True)

    X, y = _build_feature_matrix(df, target_col)

    split_cfg = {"test_size": 0.2, "shuffle": False, "random_state": random_seed}
    split_cfg.update(split_params)
    split_cfg["shuffle"] = False
    X_train, X_test, y_train, y_test = train_test_split(X, y, **split_cfg)

    try:
        import xgboost as xgb
    except ModuleNotFoundError:  # pragma: no cover - optional dependency
        xgboost_available = False
        xgb = None  # type: ignore[assignment]
    else:
        xgboost_available = True

    if xgboost_available:
        use_gpu_flag = bool(gpu and CONFIG.models.use_gpu)
        params = {
            "n_estimators": 200,
            "tree_method": "gpu_hist" if use_gpu_flag else "hist",
            "eval_metric": "logloss",
            "random_state": random_seed,
        }
        model = xgb.XGBClassifier(**params)
    else:
        from sklearn.ensemble import RandomForestClassifier

        logger.warning("xgboost not available; falling back to RandomForestClassifier.")
        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=6,
            n_jobs=-1,
            random_state=random_seed,
        )

    model.fit(X_train, y_train)

    probas = model.predict_proba(X_test)[:, 1]
    preds = (probas >= 0.5).astype(np.int8)
    accuracy = float(accuracy_score(y_test, preds))

    metrics = {"accuracy": accuracy, "test_size": split_cfg.get("test_size", 0.2)}
    with (run_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    pred_df = pd.DataFrame(
        {
            "timestamp": df.loc[y_test.index, "timestamp"].reset_index(drop=True),
            "y_true": y_test.reset_index(drop=True),
            "y_pred_proba": probas,
            "y_pred": preds,
        }
    )
    pred_df.to_csv(run_dir / "predictions_clf.csv", index=False)

    if xgboost_available:
        try:  # optional explainability artefacts
            import matplotlib.pyplot as plt
            import shap
            from sklearn.inspection import permutation_importance

            groups = assign_feature_groups(list(X.columns))

            perm = permutation_importance(
                model, X_test, y_test, n_repeats=5, random_state=random_seed
            )
            perm_df = pd.DataFrame(
                {"feature": X.columns, "importance": perm.importances_mean}
            ).sort_values("importance", ascending=False)
            perm_df.to_csv(run_dir / "perm_importance_clf.csv", index=False)

            explainer = shap.Explainer(model, X_train)
            shap_vals = explainer(X_test).values
            shap_mean = (
                np.abs(shap_vals)
                .mean(axis=0)
            )
            shap_df = pd.DataFrame(
                {"feature": X.columns, "mean_abs_shap": shap_mean}
            ).sort_values("mean_abs_shap", ascending=False)
            shap_df.to_csv(run_dir / "shap_values_clf.csv", index=False)

            shap_df["group"] = shap_df["feature"].map(groups)
            shap_group = (
                shap_df.groupby("group")["mean_abs_shap"].sum().sort_values(ascending=False)
            )
            shap_group.to_csv(run_dir / "shap_group_clf.csv")

            perm_df["group"] = perm_df["feature"].map(groups)
            perm_group = (
                perm_df.groupby("group")["importance"].sum().sort_values(ascending=False)
            )
            perm_group.to_csv(run_dir / "perm_group_clf.csv")

            fig, ax = plt.subplots()
            shap_df.head(20).plot.barh(x="feature", y="mean_abs_shap", ax=ax)
            fig.tight_layout()
            fig.savefig(run_dir / "shap_top20_clf.png")
            plt.close(fig)

            fig, ax = plt.subplots()
            perm_df.head(20).plot.barh(x="feature", y="importance", ax=ax)
            fig.tight_layout()
            fig.savefig(run_dir / "perm_top20_clf.png")
            plt.close(fig)

        except Exception:  # pragma: no cover - optional deps may be missing
            logger.info("Skipping explainability outputs due to missing dependencies")

        booster_path = run_dir / "clf_model.json"
        model.get_booster().save_model(booster_path)
    else:
        booster_path = run_dir / "clf_model.json"
        with booster_path.open("w", encoding="utf-8") as f:
            json.dump({"model": "RandomForestClassifier"}, f)

    config_snapshot = config_to_dict(CONFIG)
    with (run_dir / "config_snapshot.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(config_snapshot, f, sort_keys=False)

    metadata = {
        "run_id": run_id,
        "task": task,
        "horizon_minutes": horizon,
        "txn_cost_bps": txn_cost_bps,
        "split_params": split_params,
        "use_onchain": feature_settings.include_onchain,
        "gpu_requested": bool(gpu),
        "random_seed": random_seed,
        "config_path": str(CONFIG.config_path) if CONFIG.config_path else None,
        "created_at": datetime.utcnow().isoformat(),
        "output_dir": str(run_dir),
    }
    with (run_dir / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)


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
