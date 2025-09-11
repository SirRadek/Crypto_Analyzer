"""Helpers for meta-model inference with optimized imports."""
from __future__ import annotations

from typing import Iterable, TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    import numpy as np
    from sklearn.ensemble import RandomForestClassifier

__all__ = ["batch_predict"]


def batch_predict(model, X: "np.ndarray", *, batch_size: int = 1000):
    """Yield predictions in batches with contiguous float32 arrays."""
    import numpy as np

    for start in range(0, len(X), batch_size):
        end = start + batch_size
        Xb = np.ascontiguousarray(X[start:end], dtype=np.float32)
        yield model.predict(Xb)
