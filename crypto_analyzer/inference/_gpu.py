from __future__ import annotations

import importlib
import logging
from typing import Optional, Tuple, Type

logger = logging.getLogger(__name__)


def get_gpu_rf_or_none(use_gpu: bool = False) -> Optional[Tuple[Type[object], Type[object]]]:
    """Return GPU RandomForest classes if available.

    The heavy ``cuml`` and ``cudf`` imports are attempted only when
    ``use_gpu`` is ``True``.  When the import fails (missing packages or
    CUDA runtime), a warning is logged and ``None`` is returned.
    """

    if not use_gpu:
        return None
    try:
        cuml = importlib.import_module("cuml.ensemble")
        importlib.import_module("cudf")
        return getattr(cuml, "RandomForestClassifier"), getattr(cuml, "RandomForestRegressor")
    except Exception as exc:  # pragma: no cover - behaves same for any failure
        logger.warning("GPU RF unavailable: %s", exc)
        return None
