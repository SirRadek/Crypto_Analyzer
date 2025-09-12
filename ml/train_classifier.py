from __future__ import annotations

import argparse
import json
import logging
import traceback
from collections.abc import Sequence
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import xgboost as xgb

from crypto_analyzer.model_manager import MODELS_ROOT, PROJECT_ROOT, atomic_write

MODEL_PATH = "ml/meta_model_clf.joblib"
logger = logging.getLogger(__name__)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train classifier")
    period = parser.add_mutually_exclusive_group()
    period.add_argument("--train-window")
    period.add_argument("--train-start")
    parser.add_argument("--train-end")
    parser.add_argument("--horizon")
    parser.add_argument("--step")
    eval_group = parser.add_mutually_exclusive_group()
    eval_group.add_argument("--eval-split")
    eval_group.add_argument("--eval-frac", type=float)
    parser.add_argument("--reset-metadata", action="store_true")
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.train_start and not args.train_end:
        parser.error("--train-start requires --train-end")
    if args.train_end and not args.train_start:
        parser.error("--train-end requires --train-start")
    if args.train_window and (args.train_start or args.train_end):
        parser.error("--train-window is mutually exclusive with --train-start/--train-end")
    return args


def train_classifier(
    X,
    y,
    model_path: str = MODEL_PATH,
    *,
    params: dict[str, Any] | None = None,
    use_gpu: bool = False,
    train_window: int | None = None,
) -> xgb.XGBClassifier:
    """Train an XGBoost classifier with optional GPU acceleration."""

    if train_window is not None:
        X = X[-train_window:]
        y = y[-train_window:]

    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)

    xgb.set_config(verbosity=0)

    params = params.copy() if params else {}
    params.setdefault("tree_method", "hist")
    params.setdefault("device", "cuda" if use_gpu else "cpu")
    params.setdefault("max_depth", 6)
    params.setdefault("n_estimators", 400)
    params.setdefault("subsample", 0.8)
    params.setdefault("colsample_bytree", min(0.8, 1.0))
    params.setdefault("early_stopping_rounds", 50)
    params.setdefault("nthread", -1)
    params.setdefault("random_state", 42)
    params.setdefault("verbosity", 0)

    n_estimators = params.pop("n_estimators")
    early_rounds = params.pop("early_stopping_rounds")

    eval_size = max(1, int(0.2 * len(X)))
    X_train, X_val = X[:-eval_size], X[-eval_size:]
    y_train, y_val = y[:-eval_size], y[-eval_size:]

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)

    sk_params = {
        "max_depth": params.get("max_depth"),
        "subsample": params.get("subsample"),
        "colsample_bytree": params.get("colsample_bytree"),
        "tree_method": params.get("tree_method"),
        "device": params.get("device"),
        "n_jobs": params.get("nthread"),
        "random_state": params.get("random_state"),
        "verbosity": params.get("verbosity"),
        "n_estimators": n_estimators,
    }
    sk_params = {k: v for k, v in sk_params.items() if v is not None}

    while True:
        try:
            booster = xgb.train(
                params,
                dtrain,
                num_boost_round=n_estimators,
                evals=[(dval, "eval")],
                early_stopping_rounds=early_rounds,
            )
            model = xgb.XGBClassifier(**sk_params)
            model._Booster = booster
            model._n_features_in = X.shape[1]
            buffer = BytesIO()
            joblib.dump(model, buffer, compress=False)
            atomic_write(Path(model_path), buffer.getvalue())
            return model
        except xgb.core.XGBoostError as exc:
            if params.get("device") == "cuda":
                logger.warning("CUDA not available, falling back to CPU")
                params["device"] = "cpu"
                sk_params["device"] = "cpu"
                continue
            raise exc


def _reset_metadata() -> None:
    MODELS_ROOT.mkdir(parents=True, exist_ok=True)
    for name in ["model_usage.json", "model_performance.json"]:
        path = MODELS_ROOT / name
        atomic_write(path, b"{}")
        json.loads(path.read_text())  # validate


def _log_run(config: dict[str, object]) -> None:
    runs_dir = PROJECT_ROOT / "logs" / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    data = json.dumps({"config": config}, indent=2).encode("utf-8")
    atomic_write(runs_dir / f"run_{ts}.json", data)


def _log_failure(stem: str, exc: BaseException) -> None:
    fails = PROJECT_ROOT / "logs" / "failures.jsonl"
    fails.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "stem": stem,
        "exc_type": type(exc).__name__,
        "message": str(exc),
        "hash": hash(traceback.format_exc()),
    }
    with fails.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.reset_metadata:
        _reset_metadata()
    try:
        # Placeholder for training logic
        _log_run(vars(args))
    except Exception as exc:  # pragma: no cover - defensive
        _log_failure("classifier", exc)
        raise
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
