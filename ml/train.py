from __future__ import annotations

import json
import logging
import os
from io import BytesIO
from pathlib import Path
from typing import Any, Iterable

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split

from crypto_analyzer.model_manager import atomic_write
from ml.model_utils import evaluate_model
from utils.splitting import WalkForwardSplit

MODEL_PATH = "ml/meta_model_cls.joblib"

logger = logging.getLogger(__name__)


def _to_f32(X) -> pd.DataFrame | np.ndarray:
    if isinstance(X, pd.DataFrame):
        return X.astype("float32", copy=False)
    return np.asarray(X, dtype=np.float32)


def _fit_no_es(clf: xgb.XGBClassifier, X_train, y_train, X_val, y_val) -> None:
    """Trénink bez early stoppingu pro verze XGBoostu bez podpory ES ve sklearn wrapperu."""
    clf.fit(X_train, y_train, eval_set=[(X_val, y_val)])


def train_model(
    X,
    y,
    model_path: str = MODEL_PATH,
    tune: bool = False,  # zachováno kvůli rozhraní, netuníme zde
    test_size: float = 0.2,
    random_state: int = 42,
    use_gpu: bool = True,  # defaultně zapínáme GPU, fallback níž
    oob_tol: float | None = None,  # zachováno kvůli kompatibilitě s voláním
    oob_step: int = 50,  # nevyužito u XGBoost, ponecháno pro signaturu
    max_estimators: int = 400,  # nevyužito u XGBoost, ponecháno pro signaturu
    log_path: str = "ml/oob_cls.json",
    split: str = "holdout",
    wfs_params: dict[str, int] | None = None,
):
    """Train meta-classifier using XGBoost.

    Parameters
    ----------
    X, y:
        Training data. ``X`` may be :class:`pandas.DataFrame` or
        :class:`numpy.ndarray`.
    model_path:
        Path where the trained model will be stored.
    tune, test_size, random_state, use_gpu, oob_tol, oob_step,
    max_estimators, log_path, split, wfs_params:
        Additional parameters retained for backwards compatibility.

    Returns
    -------
    xgb.XGBClassifier
        Trained model.
    """
    base_params: dict[str, Any] = dict(
        n_estimators=400,
        max_depth=6,
        learning_rate=0.08,
        subsample=0.8,
        colsample_bytree=0.8,
        tree_method=("gpu_hist" if use_gpu else "hist"),
        predictor=("gpu_predictor" if use_gpu else "cpu_predictor"),
        n_jobs=-1,
        eval_metric="logloss",
        random_state=random_state,
        verbosity=0,
    )

    if split == "walkforward":
        if not isinstance(X, pd.DataFrame) or "timestamp" not in X.columns:
            raise ValueError("Walkforward split vyžaduje DataFrame s 'timestamp'")
        df = X.reset_index(drop=True)
        features = df.drop(columns=["timestamp"])  # training features
        y_series = pd.Series(y).reset_index(drop=True)

        params = wfs_params or {
            "train_span_days": 30,
            "test_span_days": 7,
            "step_days": 7,
            "min_train_days": 30,
        }
        splitter = WalkForwardSplit(**params)

        fold_metrics: list[dict[str, float]] = []
        preds_frames: list[pd.DataFrame] = []

        for fold, (train_idx, test_idx) in enumerate(splitter.split(df)):
            X_train = _to_f32(features.iloc[train_idx])
            y_train = y_series.iloc[train_idx]
            X_val = _to_f32(features.iloc[test_idx])
            y_val = y_series.iloc[test_idx]

            clf = xgb.XGBClassifier(**base_params)
            try:
                _fit_no_es(clf, X_train, y_train, X_val, y_val)
            except xgb.core.XGBoostError:
                logger.warning("CUDA not available or failed. Falling back to CPU.")
                clf.set_params(tree_method="hist", predictor="cpu_predictor")
                _fit_no_es(clf, X_train, y_train, X_val, y_val)

            acc, f1 = evaluate_model(clf, X_val, y_val)
            fold_metrics.append({"fold": fold, "accuracy": acc, "f1": f1})

            preds = clf.predict(X_val)
            preds_frames.append(
                pd.DataFrame(
                    {"fold": fold, "y_true": y_val, "y_pred": preds},
                    index=test_idx,
                )
            )

        # uložit metriky
        metrics_data = {"folds": fold_metrics}
        atomic_write(Path(log_path), json.dumps(metrics_data, indent=2).encode("utf-8"))

        # uložit predikce
        if preds_frames:
            preds_df = pd.concat(preds_frames).sort_index()
            pred_path = Path(log_path).with_suffix(".preds.csv")
            preds_df.to_csv(pred_path, index_label="index")

        # finální model natrénujeme na všech datech
        X_full = _to_f32(features)
        y_full = y_series
        clf = xgb.XGBClassifier(**base_params)
        try:
            _fit_no_es(clf, X_full, y_full, X_full, y_full)
        except xgb.core.XGBoostError:
            logger.warning("CUDA not available or failed. Falling back to CPU.")
            clf.set_params(tree_method="hist", predictor="cpu_predictor")
            _fit_no_es(clf, X_full, y_full, X_full, y_full)

        buffer = BytesIO()
        joblib.dump(clf, buffer)
        atomic_write(Path(model_path), buffer.getvalue())
        logger.info("Model saved to %s", model_path)
        return clf

    # --- defaultní holdout split ---------------------------------------------

    if isinstance(X, pd.DataFrame) and "timestamp" in X.columns:
        X = X.drop(columns=["timestamp"])

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # float32 a zachování DataFrame kvůli feature_names_in_
    if isinstance(X_train, pd.DataFrame):
        X_train = _to_f32(X_train)
        X_val = _to_f32(X_val)
    else:
        X_train = _to_f32(X_train)
        X_val = _to_f32(X_val)

    clf = xgb.XGBClassifier(**base_params)

    # Primární pokus na GPU → fallback CPU při chybě
    try:
        _fit_no_es(clf, X_train, y_train, X_val, y_val)
    except xgb.core.XGBoostError:
        logger.warning("CUDA not available or failed. Falling back to CPU.")
        clf.set_params(tree_method="hist", predictor="cpu_predictor")
        _fit_no_es(clf, X_train, y_train, X_val, y_val)

    # Vyhodnocení a uložení
    evaluate_model(clf, X_val, y_val)

    buffer = BytesIO()
    joblib.dump(clf, buffer)
    atomic_write(Path(model_path), buffer.getvalue())
    logger.info("Model saved to %s", model_path)
    return clf


def load_model(model_path: str = MODEL_PATH):
    """Load a model from ``model_path``.

    Parameters
    ----------
    model_path:
        Location of the saved model.

    Returns
    -------
    Any
        Loaded object.
    """

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"No model found at {model_path}")
    return joblib.load(model_path)


# ---------------------------------------------------------------------------
# New generic training interface with CLI support
# ---------------------------------------------------------------------------


def _exp_weights(timestamps: Iterable[pd.Timestamp], half_life_days: float) -> np.ndarray:
    """Compute exponential weights with given half-life in days."""
    ts = pd.to_datetime(pd.Series(list(timestamps)))
    max_ts = ts.max()
    age_days = (max_ts - ts).dt.total_seconds() / 86400.0
    return np.exp(-np.log(2) * age_days / half_life_days)


def train_xgb(
    X: pd.DataFrame,
    y: pd.Series,
    timestamps: pd.Series,
    task: str,
    output_dir: Path,
    *,
    gpu: bool = False,
    seed: int = 42,
    eval_metric: str = "logloss",
    n_estimators: int = 200,
    max_depth: int = 6,
    eta: float = 0.1,
    subsample: float = 1.0,
    colsample_bytree: float = 1.0,
    min_child_weight: float = 1.0,
    reg_lambda: float = 1.0,
    class_threshold: float = 0.5,
    scale_pos_weight: float = 1.0,
    half_life_days: float = 30.0,
) -> xgb.Booster:
    """Train XGBoost model with exponential time decay weighting."""

    use_gpu = gpu
    tree_method = "gpu_hist" if use_gpu else "hist"

    params: dict[str, Any] = {
        "objective": "binary:logistic" if task == "clf" else "reg:squarederror",
        "eval_metric": eval_metric,
        "max_depth": max_depth,
        "eta": eta,
        "subsample": subsample,
        "colsample_bytree": colsample_bytree,
        "min_child_weight": min_child_weight,
        "lambda": reg_lambda,
        "tree_method": tree_method,
        "seed": seed,
    }
    if task == "clf":
        params["scale_pos_weight"] = scale_pos_weight

    # chronological train/test split (80/20)
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    ts_train, ts_test = timestamps.iloc[:split_idx], timestamps.iloc[split_idx:]

    weights = _exp_weights(ts_train, half_life_days)

    dtrain = xgb.DMatrix(X_train, label=y_train, weight=weights)
    dtest = xgb.DMatrix(X_test, label=y_test)

    try:
        booster = xgb.train(params, dtrain, num_boost_round=n_estimators)
    except xgb.core.XGBoostError:
        if use_gpu:
            params["tree_method"] = "hist"
            booster = xgb.train(params, dtrain, num_boost_round=n_estimators)
        else:
            raise

    # Evaluation metrics
    preds = booster.predict(dtest)
    metrics: dict[str, float] = {}
    if task == "clf":
        from sklearn.metrics import (
            accuracy_score,
            f1_score,
            precision_score,
            recall_score,
            roc_auc_score,
        )

        labels = (preds >= class_threshold).astype(int)
        metrics["accuracy"] = float(accuracy_score(y_test, labels))
        metrics["f1"] = float(f1_score(y_test, labels))
        metrics["precision"] = float(precision_score(y_test, labels))
        metrics["recall"] = float(recall_score(y_test, labels))
        metrics["roc_auc"] = float(roc_auc_score(y_test, preds))
    else:
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

        metrics["rmse"] = float(np.sqrt(mean_squared_error(y_test, preds)))
        metrics["mae"] = float(mean_absolute_error(y_test, preds))
        metrics["r2"] = float(r2_score(y_test, preds))

    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv(output_dir / "metrics_cv.csv", index=False)

    # SHAP feature importance
    import shap

    explainer = shap.TreeExplainer(booster)
    sample = min(len(X_test), 20_000)
    X_sample = X_test.iloc[:sample]
    shap_vals = explainer.shap_values(X_sample)
    shap_abs = np.abs(shap_vals)
    shap_mean = shap_abs.mean(axis=0)
    fi = pd.DataFrame({"feature": X_sample.columns, "mean_abs_shap": shap_mean}).sort_values(
        "mean_abs_shap", ascending=False
    )
    fi.to_csv(output_dir / "feature_importance_shap.csv", index=False)

    # Top-20 feature plot
    top20 = fi.head(20)
    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 6))
    plt.barh(top20["feature"][::-1], top20["mean_abs_shap"][::-1])
    plt.tight_layout()
    plt.savefig(output_dir / "shap_top20.png")
    plt.close()

    # Group importance
    from analysis.feature_engineering import assign_feature_groups

    groups = assign_feature_groups(list(X_sample.columns))
    fi["group"] = fi["feature"].map(groups)
    group_imp = fi.groupby("group")["mean_abs_shap"].sum().sort_values(ascending=False)
    total = group_imp.sum()
    group_df = (
        group_imp.to_frame(name="mean_abs_shap")
        .assign(share=lambda d: d["mean_abs_shap"] / total * 100)
        .reset_index()
    )
    group_df.to_csv(output_dir / "group_importance_shap.csv", index=False)

    from reporting.report_utils import plot_group_bars

    plot_group_bars(
        group_df.rename(columns={"mean_abs_shap": "value"}),
        output_dir / "group_shap.png",
        value_col="value",
    )

    # Permutation importance in temporal blocks
    from sklearn.metrics import log_loss, mean_squared_error

    block_size = max(1, len(X_test) // 10)
    base_score = log_loss(y_test, preds) if task == "clf" else mean_squared_error(y_test, preds)
    perm_scores: dict[str, float] = {}
    rng = np.random.default_rng(seed)
    for col in X_test.columns:
        scores = []
        for _ in range(5):
            X_perm = X_test.copy()
            for start in range(0, len(X_perm), block_size):
                end = start + block_size
                idx = slice(start, end)
                block = X_perm.loc[X_perm.index[idx], col].values
                rng.shuffle(block)
                X_perm.loc[X_perm.index[idx], col] = block
            dperm = xgb.DMatrix(X_perm, label=y_test)
            perm_preds = booster.predict(dperm)
            score = (
                log_loss(y_test, perm_preds)
                if task == "clf"
                else mean_squared_error(y_test, perm_preds)
            )
            scores.append(score)
        perm_scores[col] = float(np.mean(scores) - base_score)
    perm_df = pd.DataFrame(
        {"feature": perm_scores.keys(), "importance": perm_scores.values()}
    ).sort_values("importance", ascending=False)
    perm_df.to_csv(output_dir / "permutation_importance.csv", index=False)

    # Drift computation (rolling 30d)
    shap_ts = pd.DataFrame(shap_abs, columns=X_sample.columns, index=ts_test.iloc[:sample])
    top_feats = fi.head(5)["feature"].tolist()
    drift_feat = shap_ts[top_feats].rolling("30D").mean()
    drift_feat.to_csv(output_dir / "drift_features.csv")

    group_ts = shap_ts.groupby(groups, axis=1).sum()
    top_groups = group_df.head(5)["group"].tolist()
    drift_group = group_ts[top_groups].rolling("30D").mean()
    drift_group.to_csv(output_dir / "drift_groups.csv")

    plt.figure(figsize=(8, 6))
    for col in top_groups:
        plt.plot(drift_group.index, drift_group[col], label=col)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "drift_groups.png")
    plt.close()

    return booster


def parse_args(argv: Iterable[str] | None = None):
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["clf", "reg"], required=True)
    parser.add_argument("--horizon", type=int, choices=[120, 240], required=True)
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval_metric", default="logloss")
    parser.add_argument("--n_estimators", type=int, default=200)
    parser.add_argument("--max_depth", type=int, default=6)
    parser.add_argument("--eta", type=float, default=0.1)
    parser.add_argument("--subsample", type=float, default=1.0)
    parser.add_argument("--colsample_bytree", type=float, default=1.0)
    parser.add_argument("--min_child_weight", type=float, default=1.0)
    parser.add_argument("--lambda", dest="reg_lambda", type=float, default=1.0)
    parser.add_argument("--class_threshold", type=float, default=0.5)
    parser.add_argument("--scale_pos_weight", type=float, default=1.0)
    parser.add_argument(
        "--half_life", type=float, default=30.0, help="Half-life in days for weights"
    )
    parser.add_argument("--run_id", type=str, default=None)
    return parser.parse_args(argv)


def main_cli(args) -> Path:
    from analysis.feature_engineering import FEATURE_COLUMNS, create_features
    from db.db_connector import get_price_data
    from utils.config import CONFIG

    run_id = args.run_id or pd.Timestamp.utcnow().strftime("%Y%m%d_%H%M%S")
    out_dir = Path("outputs") / run_id

    df = get_price_data(CONFIG.symbol, db_path=CONFIG.db_path)
    df = create_features(df)
    steps = args.horizon // 5
    df["target_cls"] = (df["close"].shift(-steps) > df["close"]).astype(int)
    df["target_reg"] = df["close"].shift(-steps)
    df = df.dropna(subset=["target_cls", "target_reg"])

    X = df[FEATURE_COLUMNS].astype("float32")
    y = df["target_cls" if args.task == "clf" else "target_reg"]
    timestamps = df["timestamp"]

    train_xgb(
        X,
        y,
        timestamps,
        args.task,
        out_dir,
        gpu=args.gpu,
        seed=args.seed,
        eval_metric=args.eval_metric,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        eta=args.eta,
        subsample=args.subsample,
        colsample_bytree=args.colsample_bytree,
        min_child_weight=args.min_child_weight,
        reg_lambda=args.reg_lambda,
        class_threshold=args.class_threshold,
        scale_pos_weight=args.scale_pos_weight,
        half_life_days=args.half_life,
    )
    return out_dir


if __name__ == "__main__":
    cli_args = parse_args()
    main_cli(cli_args)
