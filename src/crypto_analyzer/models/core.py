"""Unified model abstraction layer for classical estimators.

This module introduces a tiny object oriented wrapper around the models used
within the project.  The :class:`BaseModel` defines the persistence and common
training/prediction helpers while concrete implementations provide the
underlying estimator (e.g. Random Forest or XGBoost).  A lightweight
``EnsembleModel`` is also available to combine multiple base models using either
simple averaging or a small logistic regression meta-model.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable, Sequence
from io import BytesIO
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression

from crypto_analyzer.model_manager import atomic_write

try:  # pragma: no cover - optional dependency, exercised in integration tests
    import xgboost as xgb
except ModuleNotFoundError:  # pragma: no cover - handled via explicit error
    xgb = None  # type: ignore[assignment]


class BaseModel(ABC):
    """Common interface shared by all classical models used in the project."""

    def __init__(self, *, model_path: str | Path | None = None):
        self.model_path = Path(model_path) if model_path else None
        self._estimator: Any | None = None
        self._feature_names: list[str] | None = None
        self._init_params: dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Life-cycle helpers
    # ------------------------------------------------------------------

    def train(self, X: pd.DataFrame | np.ndarray, y: Iterable[Any]) -> BaseModel:
        """Fit the underlying estimator on the provided dataset."""

        features = self._prepare_features(X, record_names=True)
        target = np.asarray(y)
        estimator = self._build_estimator()
        self._fit_estimator(estimator, features, target)
        self._estimator = estimator
        return self

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Generate predictions using the trained estimator."""

        if self._estimator is None:
            raise RuntimeError("Model has not been trained yet.")
        features = self._prepare_features(X, record_names=False)
        return self._predict(self._estimator, features)

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------

    def save(self, path: str | Path | None = None) -> Path:
        """Persist the model to disk using :mod:`joblib`."""

        if self._estimator is None:
            raise RuntimeError("Cannot save an untrained model.")

        destination = Path(path or self.model_path or "")
        if not destination:
            raise ValueError("A destination path must be provided to save the model.")

        payload = self._state_dict()
        buffer = BytesIO()
        joblib.dump(payload, buffer)
        atomic_write(destination, buffer.getvalue())
        self.model_path = destination
        return destination

    @classmethod
    def load(cls, path: str | Path) -> BaseModel:
        """Restore a model instance from *path*."""

        payload = joblib.load(path)
        if not isinstance(payload, dict) or "estimator" not in payload:
            raise ValueError("Serialized payload does not describe a BaseModel.")

        init_params = payload.get("init_params", {})
        model = cls(**init_params)
        model.model_path = Path(path)
        model._load_state(payload)
        return model

    # ------------------------------------------------------------------
    # Hooks for subclasses
    # ------------------------------------------------------------------

    @abstractmethod
    def _build_estimator(self) -> Any:
        """Instantiate the underlying estimator."""

    def _fit_estimator(self, estimator: Any, X: np.ndarray, y: np.ndarray) -> None:
        estimator.fit(X, y)

    def _predict(self, estimator: Any, X: np.ndarray) -> np.ndarray:
        if hasattr(estimator, "predict_proba"):
            proba = estimator.predict_proba(X)
            if isinstance(proba, np.ndarray) and proba.ndim == 2 and proba.shape[1] > 1:
                return np.asarray(proba[:, 1])
            return np.asarray(proba).ravel()
        return np.asarray(estimator.predict(X))

    # ------------------------------------------------------------------
    # Internal utilities
    # ------------------------------------------------------------------

    def _prepare_features(self, X: pd.DataFrame | np.ndarray, *, record_names: bool) -> np.ndarray:
        if isinstance(X, pd.DataFrame):
            if record_names:
                self._feature_names = [str(c) for c in X.columns]
            return X.to_numpy(dtype=np.float32, copy=False)

        arr = np.asarray(X, dtype=np.float32)
        if record_names and self._feature_names is None and arr.ndim == 2:
            self._feature_names = [f"f{i}" for i in range(arr.shape[1])]
        return arr

    def _state_dict(self) -> dict[str, Any]:
        return {
            "estimator": self._estimator,
            "feature_names": self._feature_names,
            "init_params": self._init_params,
        }

    def _load_state(self, state: dict[str, Any]) -> None:
        self._estimator = state.get("estimator")
        self._feature_names = state.get("feature_names")

    def _set_init_params(self, **params: Any) -> None:
        self._init_params = {k: v for k, v in params.items() if k != "model_path"}


class RandomForestModel(BaseModel):
    """Wrapper around scikit-learn Random Forest models."""

    def __init__(
        self,
        *,
        n_estimators: int = 100,
        max_depth: int | None = None,
        random_state: int | None = None,
        n_jobs: int = -1,
        task: str = "classification",
        model_path: str | Path | None = None,
    ) -> None:
        super().__init__(model_path=model_path)
        task_lower = task.lower()
        if task_lower not in {"classification", "regression"}:
            raise ValueError("task must be 'classification' or 'regression'")

        self.n_estimators = int(n_estimators)
        self.max_depth = max_depth
        self.random_state = random_state
        self.n_jobs = int(n_jobs)
        self.task = task_lower
        self._set_init_params(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
            task=self.task,
        )

    def _build_estimator(self) -> Any:
        if self.task == "classification":
            return RandomForestClassifier(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                random_state=self.random_state,
                n_jobs=self.n_jobs,
            )
        return RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
        )


class XGBoostModel(BaseModel):
    """Wrapper around :class:`xgboost.XGBClassifier` with GPU fallback."""

    def __init__(
        self,
        params: dict[str, Any] | None = None,
        *,
        use_gpu: bool = True,
        model_path: str | Path | None = None,
    ) -> None:
        if xgb is None:  # pragma: no cover - dependency missing guard
            raise ModuleNotFoundError("xgboost is required to use XGBoostModel")

        super().__init__(model_path=model_path)
        self.use_gpu = use_gpu
        base_params: dict[str, Any] = dict(params or {})
        if "tree_method" not in base_params:
            base_params["tree_method"] = "gpu_hist" if use_gpu else "hist"
        if "predictor" not in base_params:
            base_params["predictor"] = "gpu_predictor" if use_gpu else "cpu_predictor"
        self.params = base_params
        self._set_init_params(params=self.params, use_gpu=self.use_gpu)

    def _build_estimator(self) -> Any:
        return xgb.XGBClassifier(**self.params)

    def _fit_estimator(self, estimator: Any, X: np.ndarray, y: np.ndarray) -> None:
        try:
            estimator.fit(X, y)
        except xgb.core.XGBoostError:
            if self.use_gpu:
                estimator.set_params(tree_method="hist", predictor="cpu_predictor")
                self.use_gpu = False
                estimator.fit(X, y)
            else:
                raise

    def _predict(self, estimator: Any, X: np.ndarray) -> np.ndarray:
        """Preserve the classifier's discrete predictions for evaluation."""

        predictions = estimator.predict(X)
        return np.asarray(predictions).ravel()


class EnsembleModel(BaseModel):
    """Aggregate predictions of multiple :class:`BaseModel` instances."""

    def __init__(
        self,
        base_models: Sequence[BaseModel] | None = None,
        *,
        strategy: str = "mean",
        model_path: str | Path | None = None,
    ) -> None:
        super().__init__(model_path=model_path)
        self.base_models: list[BaseModel] = list(base_models or [])
        strategy_lower = strategy.lower()
        if strategy_lower not in {"mean", "logistic"}:
            raise ValueError("strategy must be either 'mean' or 'logistic'")
        self.strategy = strategy_lower
        self.meta_model: LogisticRegression | None = None
        if self.strategy == "logistic":
            self.meta_model = LogisticRegression(max_iter=1000)
        self._set_init_params(strategy=self.strategy)

    def train(self, X: pd.DataFrame | np.ndarray, y: Iterable[Any]) -> EnsembleModel:
        if not self.base_models:
            raise ValueError("EnsembleModel requires at least one base model")

        features = self._prepare_features(X, record_names=True)
        target = np.asarray(y)
        stacked_preds: list[np.ndarray] = []
        for model in self.base_models:
            model.train(features, target)
            preds = np.asarray(model.predict(features)).reshape(-1, 1)
            stacked_preds.append(preds)

        meta_features = np.hstack(stacked_preds)
        if self.strategy == "logistic":
            assert self.meta_model is not None
            self.meta_model.fit(meta_features, target)
            self._estimator = self.meta_model
        else:
            # averaging does not use a meta-estimator, keep None
            self._estimator = None

        self._feature_names = [f"model_{i}" for i in range(len(self.base_models))]
        return self

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        if not self.base_models:
            raise ValueError("EnsembleModel requires at least one base model")

        features = self._prepare_features(X, record_names=False)
        stacked_preds = [
            np.asarray(model.predict(features)).reshape(-1, 1) for model in self.base_models
        ]
        meta_features = np.hstack(stacked_preds)

        if self.strategy == "logistic" and self._estimator is not None:
            estimator = self._estimator
            if hasattr(estimator, "predict_proba"):
                proba = estimator.predict_proba(meta_features)
                if proba.ndim == 2 and proba.shape[1] > 1:
                    return np.asarray(proba[:, 1])
                return np.asarray(proba).ravel()
            return np.asarray(estimator.predict(meta_features))

        return meta_features.mean(axis=1)

    def _state_dict(self) -> dict[str, Any]:
        state = super()._state_dict()
        state["base_models"] = self.base_models
        return state

    def _load_state(self, state: dict[str, Any]) -> None:
        super()._load_state(state)
        self.base_models = state.get("base_models", [])

    def _build_estimator(self) -> Any:
        return self.meta_model


__all__ = ["BaseModel", "RandomForestModel", "XGBoostModel", "EnsembleModel"]
