import joblib
import os
from typing import Optional, Dict, Any, Tuple
import numpy as np
from sklearn.ensemble import RandomForestRegressor

MODEL_PATH = "ml/model_reg.joblib"
MAX_MODEL_BYTES = 200 * 1024**3  # 200 GB guard

def _fits_size(path: str, max_bytes: int = MAX_MODEL_BYTES) -> bool:
    return os.path.exists(path) and os.path.getsize(path) <= max_bytes


try:
    from numpy.core._exceptions import _ArrayMemoryError
    _MEM_ERRORS = (MemoryError, _ArrayMemoryError)
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
    params: Optional[Dict[str, Any]] = None,
    fallback_estimators: Tuple[int, ...] = (600, 400, 200, 100)
):
    """
    Train a regressor and ensure the saved model file <= 200 GB.
    Will try a sequence of n_estimators (fallback_estimators) until size fits.
    """
    if params is None:
        params = dict(
            n_estimators=fallback_estimators[0],
            random_state=42,
            n_jobs=-1,
            min_samples_leaf=2,
            verbose=0
        )

    for n in fallback_estimators:
        params["n_estimators"] = n
        try:
            model = RandomForestRegressor(**params)
            model.fit(X, y, sample_weight=sample_weight)
            # Save uncompressed to allow memory-mapped loading later
            joblib.dump(model, model_path, compress=False)
        except _MEM_ERRORS:
            # If training or saving fails due to insufficient memory, try
            # with fewer estimators to reduce memory footprint.
            print(
                f"MemoryError with n_estimators={n}; trying fewer trees if available."
            )
            del model
            continue

        size_ok = _fits_size(model_path, MAX_MODEL_BYTES)
        size_mb = os.path.getsize(model_path) / (1024**2)
        print(
            f"Regressor saved to {model_path} (n_estimators={n}, size={size_mb:.2f} MB)"
        )
        if size_ok:
            return model
        else:
            print(
                f"Model exceeds {MAX_MODEL_BYTES/(1024**3):.0f} GB, retrying with fewer trees..."
            )

    raise RuntimeError(
        "Could not save model under the size limit or training failed due to memory constraints."
    )

def load_regressor(model_path: str = MODEL_PATH, mmap_mode: Optional[str] = None):
    """Load a previously trained regressor with graceful OOM handling."""

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"No regressor at {model_path}")

    mode = mmap_mode
    if mode is None and _should_mmap(model_path):
        mode = "r"

    try:
        return joblib.load(model_path, mmap_mode=mode)
    except _MEM_ERRORS:
        if mode != "r":
            print("Memory low; retrying regressor load with mmap")
            return joblib.load(model_path, mmap_mode="r")
        raise
