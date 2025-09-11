import logging
import os
from io import BytesIO
from pathlib import Path
from typing import Any

import joblib
from sklearn.ensemble import RandomForestRegressor

from .oob import fit_incremental_forest, halving_random_search
from .train import _gpu_available
from crypto_analyzer.model_manager import atomic_write

MODEL_PATH = "ml/meta_model_reg.joblib"
MAX_MODEL_BYTES = 100 * 1024**3  # 200 GB guard


logger = logging.getLogger(__name__)


def _fits_size(path: str, max_bytes: int = MAX_MODEL_BYTES) -> bool:
    return os.path.exists(path) and os.path.getsize(path) <= max_bytes


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
    fallback_estimators: tuple[int, ...] = (600, 400, 200, 100),
    use_gpu: bool = False,
    tune: bool = False,
    oob_tol: float | None = None,
    oob_step: int = 50,
    max_estimators: int = 400,
    log_path: str = "ml/oob_reg.json",
):
    """
    Train a regressor and ensure the saved model file <= 200 GB.
    Will try a sequence of n_estimators (fallback_estimators) until size fits.

    Parameters
    ----------
    use_gpu : bool
        If ``True`` and CUDA with ``cuml`` is available, train a GPU RandomForest.
        Otherwise, a CPU-based ``RandomForestRegressor`` is used.
    """
    if use_gpu:
        if _gpu_available():
            try:  # pragma: no cover - optional dependency
                import cudf  # type: ignore
                from cuml.ensemble import RandomForestRegressor as cuRF  # type: ignore
            except Exception:  # pragma: no cover - optional dependency
                logger.warning("cuml not available, falling back to CPU")
            else:
                params_gpu = params or dict(n_estimators=fallback_estimators[0], random_state=42)
                model = cuRF(**params_gpu)
                X_f = cudf.from_pandas(X.astype("float32"))
                y_f = cudf.Series(y.astype("float32"))
                model.fit(X_f, y_f, sample_weight=sample_weight)
                buffer = BytesIO()
                joblib.dump(model, buffer, compress=False)
                atomic_write(Path(model_path), buffer.getvalue())
                return model
        else:
            logger.warning("CUDA not available, falling back to CPU")

    if tune:
        tuned = halving_random_search(X, y, RandomForestRegressor, random_state=42)
        params = {**(params or {}), **tuned}

    if oob_tol is not None:
        model, oob_scores = fit_incremental_forest(
            X,
            y,
            RandomForestRegressor,
            step=oob_step,
            max_estimators=max_estimators,
            tol=oob_tol,
            random_state=42,
            log_path=log_path,
            **(params or {}),
        )
        buffer = BytesIO()
        joblib.dump(model, buffer, compress=False)
        atomic_write(Path(model_path), buffer.getvalue())
        return model

    if params is None:
        params = dict(
            n_estimators=fallback_estimators[0],
            random_state=42,
            n_jobs=-1,
            min_samples_leaf=2,
            verbose=0,
        )

    for n in fallback_estimators:
        params["n_estimators"] = n
        model = RandomForestRegressor(**params)
        model.fit(X, y, sample_weight=sample_weight)
        buffer = BytesIO()
        joblib.dump(model, buffer, compress=False)
        atomic_write(Path(model_path), buffer.getvalue())

        size_ok = _fits_size(model_path, MAX_MODEL_BYTES)
        size_mb = os.path.getsize(model_path) / (1024**2)
        print(f"Regressor saved to {model_path} (n_estimators={n}, size={size_mb:.2f} MB)")
        if size_ok:
            return model
        else:
            print(f"Model exceeds {MAX_MODEL_BYTES/(1024**3):.0f} GB, retrying with fewer trees...")

    raise RuntimeError(
        "Could not save model under the size limit; try reducing complexity further."
    )


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


import argparse
import json
import traceback
from datetime import datetime
from typing import Iterable

from crypto_analyzer.model_manager import MODELS_ROOT, PROJECT_ROOT


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


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
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


def main(argv: Iterable[str] | None = None) -> int:
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
