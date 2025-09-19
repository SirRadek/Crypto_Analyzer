"""Typed XGBoost wrapper used alongside :class:`~crypto_analyzer.models.model_lstm.ModelLSTM`."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict

try:  # pragma: no cover - optional dependency
    import xgboost as xgb
except ModuleNotFoundError as exc:  # pragma: no cover - handled dynamically
    xgb = None  # type: ignore[assignment]
    _XGB_IMPORT_ERROR = exc
else:  # pragma: no cover - import has side effects we do not test
    _XGB_IMPORT_ERROR = None

__all__ = ["XGBoostConfig", "ModelXGBoost"]


@dataclass(slots=True)
class XGBoostConfig:
    """Configuration helper mirroring the :class:`xgboost.XGBClassifier` kwargs."""

    n_estimators: int = 200
    max_depth: int = 6
    learning_rate: float = 0.1
    subsample: float = 1.0
    colsample_bytree: float = 1.0
    eval_metric: str = "logloss"
    tree_method: str = "hist"
    predictor: str = "cpu_predictor"
    n_jobs: int = -1
    random_state: int = 42
    verbosity: int = 0

    def as_kwargs(self) -> Dict[str, Any]:
        """Return a shallow copy of the configuration as keyword arguments."""

        return asdict(self)


if xgb is not None:  # pragma: no branch - executed when dependency available

    class ModelXGBoost(xgb.XGBClassifier):
        """Thin subclass wiring :class:`XGBoostConfig` into the constructor."""

        def __init__(self, config: XGBoostConfig):
            super().__init__(**config.as_kwargs())

else:  # pragma: no cover - fallback only hit when optional dependency missing

    class ModelXGBoost:  # type: ignore[no-redef]
        """Fallback object raising a clear error when XGBoost is unavailable."""

        def __init__(self, config: XGBoostConfig):  # pragma: no cover - trivial
            raise ModuleNotFoundError(
                "xgboost is required to instantiate ModelXGBoost"
            ) from _XGB_IMPORT_ERROR
