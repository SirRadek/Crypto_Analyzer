import json
import logging
import os
from io import BytesIO
from pathlib import Path
from typing import Any, cast

import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from ml.model_utils import evaluate_model
from crypto_analyzer.model_manager import atomic_write

from .oob import fit_incremental_forest, halving_random_search

MODEL_PATH = "ml/meta_model_cls.joblib"


logger = logging.getLogger(__name__)


def _gpu_available() -> bool:
    """Return ``True`` if a CUDA device is available."""
    try:  # pragma: no cover - optional dependency
        import cupy  # type: ignore

        return cupy.cuda.runtime.getDeviceCount() > 0
    except Exception:
        return False


def train_model(
    X,
    y,
    model_path: str = MODEL_PATH,
    tune: bool = False,
    test_size: float = 0.2,
    random_state: int = 42,
    use_gpu: bool = False,
    oob_tol: float | None = None,
    oob_step: int = 50,
    max_estimators: int = 400,
    log_path: str = "ml/oob_cls.json",
):
    """Train a classification model.

    Parameters
    ----------
    X, y : array-like
        Training features and labels.
    model_path : str
        Where to persist the trained model.
    tune : bool
        If True, perform GridSearchCV over ``param_grid`` to tune
        hyperparameters for higher accuracy.
    param_grid : dict, optional
        Parameter grid for GridSearchCV. A reasonable default grid is used
        when ``None``.
    test_size : float
        Fraction of data to use for validation during training.
    random_state : int
        Reproducibility seed.
    use_gpu : bool
        If ``True`` and CUDA with ``cuml`` is available, train a GPU RandomForest.
        Otherwise, silently fall back to the CPU implementation.
    """

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    if use_gpu:
        if _gpu_available():
            try:  # pragma: no cover - optional dependency
                import cudf  # type: ignore
                from cuml.ensemble import RandomForestClassifier as cuRF  # type: ignore
            except Exception:  # pragma: no cover - optional dependency
                logger.warning("cuml not available, falling back to CPU")
            else:
                X_train_f = cudf.from_pandas(X_train.astype("float32"))
                y_train_f = cudf.Series(y_train.astype("float32"))
                clf = cuRF(random_state=random_state)
                clf.fit(X_train_f, y_train_f)
                X_val_f = cudf.from_pandas(X_val.astype("float32"))
                evaluate_model(clf, X_val_f, cudf.Series(y_val.astype("float32")))
                buffer = BytesIO()
                joblib.dump(clf, buffer)
                atomic_write(Path(model_path), buffer.getvalue())
                return clf
        else:
            logger.warning("CUDA not available, falling back to CPU")

    params: dict[str, Any] = {}
    if tune:
        params = halving_random_search(X_train, y_train, RandomForestClassifier, random_state)

    if oob_tol is not None:
        clf, oob_scores = fit_incremental_forest(
            X_train,
            y_train,
            RandomForestClassifier,
            step=oob_step,
            max_estimators=max_estimators,
            tol=oob_tol,
            random_state=random_state,
            log_path=log_path,
            **params,
        )
    else:
        n_estimators = cast(int, params.pop("n_estimators", 200))
        clf = RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=random_state,
            n_jobs=-1,
            oob_score=True,
            **params,
        )
        clf.fit(X_train, y_train)
        log_bytes = json.dumps({"params": clf.get_params(), "oob_scores": [float(clf.oob_score_)]}).encode("utf-8")
        atomic_write(Path(log_path), log_bytes)

    # Evaluate on validation data
    evaluate_model(clf, X_val, y_val)

    buffer = BytesIO()
    joblib.dump(clf, buffer)
    atomic_write(Path(model_path), buffer.getvalue())
    print(f"Model saved to {model_path}")
    return clf


def load_model(model_path=MODEL_PATH):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"No model found at {model_path}")
    return joblib.load(model_path)
