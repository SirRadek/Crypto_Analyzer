import joblib
import os
from typing import Optional, Dict, Any, Tuple

from sklearn.ensemble import RandomForestRegressor

MODEL_PATH = "ml/model_reg.pkl"
MAX_MODEL_BYTES = 200 * 1024**3  # 200 GB guard

def _fits_size(path: str, max_bytes: int = MAX_MODEL_BYTES) -> bool:
    return os.path.exists(path) and os.path.getsize(path) <= max_bytes

def train_regressor(
    X,
    y,
    model_path: str = MODEL_PATH,
    sample_weight=None,
    params: Optional[Dict[str, Any]] = None,
    compress: int = 3,                 # joblib compression (0..9); 3 = fast+small
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
        model = RandomForestRegressor(**params)
        model.fit(X, y, sample_weight=sample_weight)
        joblib.dump(model, model_path, compress=compress)

        size_ok = _fits_size(model_path, MAX_MODEL_BYTES)
        size_mb = os.path.getsize(model_path) / (1024**2)
        print(f"Regressor saved to {model_path} (n_estimators={n}, size={size_mb:.2f} MB)")
        if size_ok:
            return model
        else:
            print(f"Model exceeds {MAX_MODEL_BYTES/(1024**3):.0f} GB, retrying with fewer trees...")

    raise RuntimeError("Could not save model under the size limit; try reducing complexity further.")

def load_regressor(model_path: str = MODEL_PATH):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"No regressor at {model_path}")
    return joblib.load(model_path)
