import argparse
import gc
import json
import logging
import math
import os
import traceback
from collections.abc import Sequence
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from xgboost import XGBRegressor

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
    """Train an XGBoost regressor with conservative defaults and OOM handling.

    The training data can be limited to the most recent ``train_window`` rows.
    Inputs are converted to ``float32`` to reduce memory footprint. If ``use_gpu``
    is ``True`` and a CUDA device is available, the model is trained with
    ``tree_method='gpu_hist'``; otherwise ``'hist'`` is used. When a
    ``MemoryError`` occurs, the function retries with ``n_jobs=1`` and, if
    necessary, falls back to the CPU ``hist`` tree method.
    """

    if train_window is not None:
        X = X[-train_window:]
        y = y[-train_window:]

    X = X.astype("float32") if hasattr(X, "astype") else np.asarray(X, dtype=np.float32)
    y = y.astype("float32") if hasattr(y, "astype") else np.asarray(y, dtype=np.float32)

    n_feat = X.shape[1]
    colsample = min(1.0, math.sqrt(n_feat) / n_feat)

    params = params.copy() if params else {}
    params.setdefault("n_estimators", 200)
    params.setdefault("subsample", 0.8)
    params.setdefault("max_depth", 16)
    params["max_depth"] = min(params["max_depth"], 16)
    params.setdefault("colsample_bytree", colsample)
    params.setdefault("n_jobs", -1)
    params.setdefault("random_state", 42)
    params.setdefault("verbosity", 0)

    if use_gpu and _gpu_available():
        params.setdefault("tree_method", "gpu_hist")
        params.setdefault("predictor", "gpu_predictor")
    else:
        if use_gpu:
            logger.warning("CUDA not available, falling back to CPU")
        params.setdefault("tree_method", "hist")

    attempts = [
        (params["n_jobs"], params["tree_method"]),
    ]
    if params["n_jobs"] != 1:
        attempts.append((1, params["tree_method"]))
    if params["tree_method"] != "hist":
        attempts.append((1, "hist"))

    last_exc: BaseException | None = None
    for n_jobs, tree_method in attempts:
        params["n_jobs"] = n_jobs
        params["tree_method"] = tree_method
        if tree_method == "hist":
            params.pop("predictor", None)
        model = XGBRegressor(**params)
        try:
            model.fit(X, y, sample_weight=sample_weight)
        except _MEM_ERRORS as exc:  # pragma: no cover - depends on resources
            last_exc = exc
            gc.collect()
            continue
        buffer = BytesIO()
        joblib.dump(model, buffer, compress=False)
        atomic_write(Path(model_path), buffer.getvalue())
        return model

    if last_exc is not None:
        raise last_exc
    raise RuntimeError("Training failed without a specific exception")


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
