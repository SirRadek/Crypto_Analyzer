"""Backtesting evaluation engine with pluggable metrics and split strategies."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from importlib import import_module
from importlib.util import find_spec
from typing import TYPE_CHECKING, TypedDict

import numpy as np
from sklearn.base import clone

from crypto_analyzer.eval.metrics import METRICS, MetricFunc

if TYPE_CHECKING:  # pragma: no cover - typing only
    from crypto_analyzer.utils.config import AppConfig


class EvaluationSummary(TypedDict, total=False):
    """Typed dictionary describing the evaluation report returned by the engine."""

    metrics: dict[str, float]
    details: list[dict[str, float]]


@dataclass(frozen=True)
class _MetricDefinition:
    name: str
    func: MetricFunc


def _slice_frame(data: object, start: int, end: int):
    if hasattr(data, "iloc"):
        return data.iloc[start:end]
    return data[start:end]


@dataclass(frozen=True)
class _BacktestSettings:
    metrics: tuple[str, ...]
    validation_fraction: float
    walkforward_window_days: int


_DEFAULT_SETTINGS = _BacktestSettings(
    metrics=("accuracy", "precision", "recall"),
    validation_fraction=0.2,
    walkforward_window_days=30,
)


def _resolve_backtest_settings(config: AppConfig | None) -> _BacktestSettings:
    if config is not None:
        backtest_cfg = config.backtest
        return _BacktestSettings(
            metrics=tuple(backtest_cfg.metrics),
            validation_fraction=float(backtest_cfg.validation_fraction),
            walkforward_window_days=int(backtest_cfg.walkforward_window_days),
        )

    if find_spec("pydantic") is None:
        return _DEFAULT_SETTINGS
    if find_spec("crypto_analyzer.utils.config") is None:
        return _DEFAULT_SETTINGS

    module = import_module("crypto_analyzer.utils.config")
    backtest_cfg = module.CONFIG.backtest  # type: ignore[attr-defined]
    return _BacktestSettings(
        metrics=tuple(backtest_cfg.metrics),
        validation_fraction=float(backtest_cfg.validation_fraction),
        walkforward_window_days=int(backtest_cfg.walkforward_window_days),
    )


class BacktestEngine:
    """Evaluate models using configurable holdout or walk-forward strategies."""

    def __init__(
        self,
        mode: str,
        *,
        metrics: Iterable[str] | None = None,
        validation_fraction: float | None = None,
        walkforward_window: int | None = None,
        config: AppConfig | None = None,
    ) -> None:
        backtest_cfg = _resolve_backtest_settings(config)
        self.mode = mode.lower()
        if self.mode not in {"holdout", "walkforward"}:
            raise ValueError("mode must be either 'holdout' or 'walkforward'")

        metric_names = tuple(metrics) if metrics is not None else backtest_cfg.metrics
        if not metric_names:
            raise ValueError("At least one metric must be specified")
        metric_defs: list[_MetricDefinition] = []
        for name in metric_names:
            try:
                func = METRICS[name]
            except KeyError as exc:  # pragma: no cover - defensive branch
                raise ValueError(f"Unknown metric '{name}'") from exc
            metric_defs.append(_MetricDefinition(name=name, func=func))
        self._metric_defs = tuple(metric_defs)

        if validation_fraction is None:
            validation_fraction = float(backtest_cfg.validation_fraction)
        if walkforward_window is None:
            walkforward_window = int(backtest_cfg.walkforward_window_days)

        if validation_fraction <= 0.0 or validation_fraction >= 1.0:
            raise ValueError("validation_fraction must be between 0 and 1 (exclusive)")
        if walkforward_window <= 0:
            raise ValueError("walkforward_window must be a positive integer")

        self.validation_fraction = validation_fraction
        self.walkforward_window = walkforward_window

    def run(self, model, X, y) -> EvaluationSummary:
        """Run the configured evaluation strategy for *model* on ``(X, y)``."""

        if self.mode == "holdout":
            metrics = self._run_holdout(model, X, y)
            return {"metrics": metrics, "details": [metrics]}
        return self._run_walkforward(model, X, y)

    def evaluate(self, model, X, y) -> dict[str, float]:
        """Convenience wrapper returning only the aggregated metrics dictionary."""

        summary = self.run(model, X, y)
        return summary.get("metrics", {})

    def _run_holdout(self, model, X, y) -> dict[str, float]:
        n_samples = len(y)
        if n_samples < 2:
            raise ValueError("Holdout evaluation requires at least two samples")
        validation_size = max(1, int(round(n_samples * self.validation_fraction)))
        train_end = n_samples - validation_size
        if train_end <= 0:
            raise ValueError("Not enough samples for the requested validation fraction")

        X_train = _slice_frame(X, 0, train_end)
        y_train = _slice_frame(y, 0, train_end)
        X_val = _slice_frame(X, train_end, n_samples)
        y_val = _slice_frame(y, train_end, n_samples)

        estimator = clone(model)
        estimator.fit(X_train, y_train)
        y_pred, y_proba = self._predict(estimator, X_val)
        return self._compute_metrics(y_val, y_pred, y_proba)

    def _run_walkforward(self, model, X, y) -> EvaluationSummary:
        n_samples = len(y)
        window = self.walkforward_window
        if n_samples <= window:
            raise ValueError("Walk-forward evaluation requires more samples than the window size")

        fold_metrics: list[dict[str, float]] = []
        train_end = window
        while train_end < n_samples:
            test_end = min(train_end + window, n_samples)
            X_train = _slice_frame(X, 0, train_end)
            y_train = _slice_frame(y, 0, train_end)
            X_test = _slice_frame(X, train_end, test_end)
            y_test = _slice_frame(y, train_end, test_end)
            if len(y_test) == 0:
                break

            estimator = clone(model)
            estimator.fit(X_train, y_train)
            y_pred, y_proba = self._predict(estimator, X_test)
            fold_metrics.append(self._compute_metrics(y_test, y_pred, y_proba))
            if test_end == n_samples:
                break
            train_end = test_end

        if not fold_metrics:
            raise ValueError("Walk-forward evaluation did not generate any folds")

        aggregate = {
            definition.name: float(np.mean([fold[definition.name] for fold in fold_metrics]))
            for definition in self._metric_defs
        }
        return {"metrics": aggregate, "details": fold_metrics}

    def _predict(self, estimator, X):
        y_pred = estimator.predict(X)
        if hasattr(estimator, "predict_proba"):
            proba = estimator.predict_proba(X)
            y_proba = np.asarray(proba)
        else:
            y_proba = None
        return np.asarray(y_pred), y_proba

    def _compute_metrics(
        self,
        y_true,
        y_pred: np.ndarray,
        y_proba: np.ndarray | None,
    ) -> dict[str, float]:
        y_true_arr = np.asarray(y_true)
        metric_values: dict[str, float] = {}
        for definition in self._metric_defs:
            metric_values[definition.name] = definition.func(y_true_arr, y_pred, y_proba)
        return metric_values


__all__ = ["BacktestEngine", "EvaluationSummary"]
