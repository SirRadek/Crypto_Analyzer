import argparse
import gc
import json
import logging
import math
import os
import resource
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

from .train import _gpu_available

MODEL_PATH = "ml/meta_model_reg.joblib"
logger = logging.getLogger(__name__)


try:
    from numpy.core._exceptions import _ArrayMemoryError

    _MEM_ERRORS: tuple[type[BaseException], ...] = (MemoryError, _ArrayMemoryError)
except Exception:  # pragma: no cover - numpy versions <1.20
    _MEM_ERRORS = (MemoryError,)


def _should_mmap(path: str) -> bool:
    """Decide whether to use memory-mapped loading based on available RAM."""
    try:
        import psutil

        size = os.path.getsize(path)
        avail = psutil.virtual_memory().available
        return size > 0.5 * avail
    except Exception:
        return False


def train_regressor(
    X,
    y,
    model_path: str = MODEL_PATH,
    sample_weight=None,
    params: dict[str, Any] | None = None,
    use_gpu: bool = False,
    train_window: int | None = None,
):
    """Train an XGBoost regressor with deterministic defaults and OOM fallback.

    The function always casts inputs to ``float32`` and constructs ``xgb.DMatrix``
    objects to minimise RAM usage. Default hyperparameters favour stability and
    mirror the user's specification. On ``MemoryError`` the training is retried
    according to the sequence:

    ``GPU → CPU n_jobs=1 → reduce n_estimators by 30% → max_depth −2``.

    Each fallback logs the reason alongside the current peak RSS.
    """

    if train_window is not None:
        X = X[-train_window:]
        y = y[-train_window:]
        if sample_weight is not None:
            sample_weight = sample_weight[-train_window:]

    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)
    if sample_weight is not None:
        sample_weight = np.asarray(sample_weight, dtype=np.float32)

    xgb.set_config(verbosity=0)

    n_feat = X.shape[1]
    colsample_default = min(1.0, math.sqrt(n_feat) / n_feat)

    params = params.copy() if params else {}
    params.setdefault("tree_method", "hist")
    params.setdefault("max_depth", 8)
    params.setdefault("n_estimators", 600)
    params.setdefault("subsample", 0.8)
    params.setdefault("colsample_bytree", min(0.8, colsample_default))
    params.setdefault("early_stopping_rounds", 50)
    params.setdefault("nthread", -1)
    params.setdefault("random_state", 42)
    params.setdefault("verbosity", 0)

    if use_gpu and _gpu_available():
        params["tree_method"] = "gpu_hist"
        params["predictor"] = "gpu_predictor"
    elif use_gpu:
        logger.warning("CUDA not available, falling back to CPU")
        params["tree_method"] = "hist"

    n_estimators = params.pop("n_estimators")
    early_rounds = params.pop("early_stopping_rounds")

    # evaluation split for early stopping
    eval_size = max(1, int(0.2 * len(X)))
    X_train, X_val = X[:-eval_size], X[-eval_size:]
    y_train, y_val = y[:-eval_size], y[-eval_size:]
    if sample_weight is not None:
        sw_train, sw_val = sample_weight[:-eval_size], sample_weight[-eval_size:]
    else:
        sw_train = sw_val = None
    dtrain = xgb.DMatrix(X_train, label=y_train, weight=sw_train)
    deval = xgb.DMatrix(X_val, label=y_val, weight=sw_val)

    # parameters for the sklearn wrapper used only for returning the model
    sk_params = {
        "max_depth": params.get("max_depth"),
        "subsample": params.get("subsample"),
        "colsample_bytree": params.get("colsample_bytree"),
        "tree_method": params.get("tree_method"),
        "n_jobs": params.get("nthread"),
        "random_state": params.get("random_state"),
        "predictor": params.get("predictor", None),
        "verbosity": params.get("verbosity"),
        "n_estimators": n_estimators,
    }
    # remove None values
    sk_params = {k: v for k, v in sk_params.items() if v is not None}

    reduced_estimators = False
    while True:
        try:
            booster = xgb.train(
                params,
                dtrain,
                num_boost_round=n_estimators,
                evals=[(deval, "eval")],
                early_stopping_rounds=early_rounds,
            )
            model = xgb.XGBRegressor(**sk_params)
            model._Booster = booster
            model._n_features_in = X.shape[1]
            buffer = BytesIO()
            joblib.dump(model, buffer, compress=False)
            atomic_write(Path(model_path), buffer.getvalue())
            return model
        except _MEM_ERRORS as exc:  # pragma: no cover - depends on resources
            peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
            if params.get("tree_method") == "gpu_hist":
                logger.warning("OOM on GPU, falling back to CPU n_jobs=1; peak RSS=%.0fMB", peak)
                params["tree_method"] = "hist"
                params.pop("predictor", None)
                params["nthread"] = 1
                sk_params["tree_method"] = "hist"
                sk_params.pop("predictor", None)
                sk_params["n_jobs"] = 1
            elif params.get("nthread", -1) != 1:
                logger.warning(
                    "OOM with n_jobs=%s, retrying with n_jobs=1; peak RSS=%.0fMB",
                    params.get("nthread"),
                    peak,
                )
                params["nthread"] = 1
                sk_params["n_jobs"] = 1
            elif not reduced_estimators and n_estimators > 1:
                new_estimators = max(1, int(n_estimators * 0.7))
                logger.warning(
                    "OOM; reducing n_estimators %d→%d; peak RSS=%.0fMB",
                    n_estimators,
                    new_estimators,
                    peak,
                )
                n_estimators = new_estimators
                sk_params["n_estimators"] = n_estimators
                reduced_estimators = True
            elif params.get("max_depth", 0) > 1:
                new_depth = max(1, params["max_depth"] - 2)
                logger.warning(
                    "OOM; reducing max_depth %d→%d; peak RSS=%.0fMB",
                    params["max_depth"],
                    new_depth,
                    peak,
                )
                params["max_depth"] = new_depth
                sk_params["max_depth"] = new_depth
            else:
                raise exc
            gc.collect()
            continue


def load_regressor(model_path: str = MODEL_PATH, mmap_mode: str | None = None):
    """Load a previously trained regressor with graceful OOM handling."""

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"No regressor at {model_path}")

    mode = mmap_mode
    if mode is None and _should_mmap(model_path):
        mode = "r"

    try:
        return joblib.load(model_path, mmap_mode=mode)  # type: ignore[arg-type]
    except _MEM_ERRORS:
        if mode != "r":
            print("Memory low; retrying regressor load with mmap")
            return joblib.load(model_path, mmap_mode="r")
        raise


# ---------------------------------------------------------------------------
# CLI helpers
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train regressor")
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


def _reset_metadata() -> None:
    MODELS_ROOT.mkdir(parents=True, exist_ok=True)
    for name in ["model_usage.json", "model_performance.json"]:
        path = MODELS_ROOT / name
        atomic_write(path, b"{}")
        json.loads(path.read_text())


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
        _log_run(vars(args))
    except Exception as exc:  # pragma: no cover
        _log_failure("regressor", exc)
        raise
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
