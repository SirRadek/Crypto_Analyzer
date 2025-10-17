"""Utility helpers for training and evaluating daily ML models.

This module provides a thin convenience wrapper around the scikit-learn
``RandomForest`` estimators that we use for quick experimentation on the daily
feature set.  The helper ensures that newly introduced features—most notably
``funding_rate`` and ``active_addresses``—are always considered during
training, performs a simple chronological train/test split and prints a compact
summary of the model performance so analysts can immediately validate the
impact of the additional inputs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split

NEW_DAILY_FEATURES: tuple[str, ...] = ("funding_rate", "active_addresses")


@dataclass(slots=True)
class RandomForestTrainingResult:
    """Container returned by :func:`train_daily_random_forest`."""

    model: RandomForestRegressor | RandomForestClassifier
    feature_columns: list[str]
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    predictions: pd.Series
    metrics: dict[str, float]
    prediction_frame: pd.DataFrame
    feature_importances: pd.Series


def _normalise_feature_columns(columns: Iterable[str]) -> list[str]:
    """Return an ordered list of unique feature column names."""

    seen: set[str] = set()
    ordered: list[str] = []
    for column in columns:
        if column in seen:
            continue
        ordered.append(column)
        seen.add(column)
    return ordered


def _infer_feature_columns(
    df: pd.DataFrame,
    target_col: str,
    additional: Sequence[str] | None = None,
) -> list[str]:
    """Infer usable feature columns from *df* while ensuring required columns."""

    non_features = {target_col, "timestamp", "date", "symbol", "asset"}
    inferred = [col for col in df.columns if col not in non_features]

    required: list[str] = list(NEW_DAILY_FEATURES)
    if additional:
        required.extend(additional)

    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(
            "Missing required feature columns in the provided dataframe: "
            + ", ".join(missing)
        )

    inferred.extend(required)
    return _normalise_feature_columns(inferred)


def _select_estimator(
    task: str, random_state: int, n_estimators: int, max_depth: int | None, n_jobs: int
) -> RandomForestRegressor | RandomForestClassifier:
    task_lower = task.lower()
    if task_lower not in {"regression", "classification"}:
        raise ValueError("task must be either 'regression' or 'classification'")

    if task_lower == "classification":
        return RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            n_jobs=n_jobs,
            random_state=random_state,
        )
    return RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        n_jobs=n_jobs,
        random_state=random_state,
    )


def train_daily_random_forest(
    df: pd.DataFrame,
    *,
    target_col: str,
    feature_cols: Sequence[str] | None = None,
    task: str = "regression",
    test_size: float = 0.2,
    random_state: int = 42,
    n_estimators: int = 300,
    max_depth: int | None = None,
    n_jobs: int = -1,
    shuffle: bool = False,
) -> RandomForestTrainingResult:
    """Train a Random Forest on the supplied daily feature dataframe.

    Parameters
    ----------
    df:
        DataFrame containing the engineered daily features and target column.
    target_col:
        Name of the target column to predict. The column must be present in *df*.
    feature_cols:
        Optional explicit feature column list. When ``None`` the helper infers all
        numeric columns except the target and metadata columns. Regardless of the
        provided list the ``funding_rate`` and ``active_addresses`` features are
        always appended to ensure the new signals are incorporated.
    task:
        Either ``"regression"`` or ``"classification"``.
    test_size, random_state, n_estimators, max_depth, n_jobs, shuffle:
        Standard scikit-learn hyper-parameters used for ``train_test_split`` and the
        Random Forest estimator.
    """

    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not present in dataframe")

    if feature_cols is None:
        features = _infer_feature_columns(df, target_col)
    else:
        features = _normalise_feature_columns(list(feature_cols) + list(NEW_DAILY_FEATURES))
        missing = [column for column in features if column not in df.columns]
        if missing:
            raise ValueError(
                "Feature columns requested but missing in dataframe: "
                + ", ".join(missing)
            )

    data = df.dropna(subset=features + [target_col])
    if data.empty:
        raise ValueError("No data available after dropping rows with missing values")

    X = data.loc[:, features].astype(np.float32)
    y = data.loc[:, target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        shuffle=shuffle,
    )

    estimator = _select_estimator(task, random_state, n_estimators, max_depth, n_jobs)
    estimator.fit(X_train, y_train)

    predictions = pd.Series(estimator.predict(X_test), index=y_test.index, name="prediction")

    metrics: dict[str, float]
    if task.lower() == "classification":
        metrics = {
            "accuracy": float(accuracy_score(y_test, predictions)),
        }
        print("Classification report:\n", classification_report(y_test, predictions))
    else:
        mae = mean_absolute_error(y_test, predictions)
        rmse = float(np.sqrt(mean_squared_error(y_test, predictions)))
        r2 = r2_score(y_test, predictions)
        metrics = {"mae": float(mae), "rmse": rmse, "r2": float(r2)}
        print(
            "Regression metrics:\n"
            f"  MAE : {mae:.4f}\n"
            f"  RMSE: {rmse:.4f}\n"
            f"  R^2 : {r2:.4f}"
        )

    prediction_frame = pd.DataFrame({"actual": y_test, "predicted": predictions})
    print("Test set predictions vs actual values (first 10 rows):")
    print(prediction_frame.head(10))

    importances = pd.Series(estimator.feature_importances_, index=features).sort_values(ascending=False)
    print("Random forest feature importances:")
    print(importances.to_string())

    return RandomForestTrainingResult(
        model=estimator,
        feature_columns=features,
        X_train=X_train,
        X_test=X_test,
        y_train=pd.Series(y_train, index=X_train.index, name=target_col),
        y_test=pd.Series(y_test, name=target_col),
        predictions=predictions,
        metrics=metrics,
        prediction_frame=prediction_frame,
        feature_importances=importances,
    )


__all__ = ["NEW_DAILY_FEATURES", "RandomForestTrainingResult", "train_daily_random_forest"]

