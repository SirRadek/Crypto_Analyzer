import os
from typing import Optional, Dict

import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split

from ml.model_utils import evaluate_model

MODEL_PATH = "ml/model.joblib"

def train_model(
    X,
    y,
    model_path: str = MODEL_PATH,
    tune: bool = False,
    param_grid: Optional[Dict] = None,
    test_size: float = 0.2,
    random_state: int = 42,
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
    """

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    if tune:
        param_grid = param_grid or {
            "n_estimators": [100, 200, 300],
            "max_depth": [None, 10, 20],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
        }
        base_clf = RandomForestClassifier(
            n_jobs=-1, random_state=random_state, class_weight="balanced"
        )
        search = GridSearchCV(
            base_clf,
            param_grid=param_grid,
            cv=5,
            n_jobs=-1,
            scoring="accuracy",
            verbose=2,
        )
        search.fit(X_train, y_train)
        clf = search.best_estimator_
        print("Best params:", search.best_params_)
    else:
        clf = RandomForestClassifier(
            n_estimators=200,
            random_state=random_state,
            n_jobs=-1,
            min_samples_leaf=2,
            verbose=1,  # <- prints training progress
        )
        clf.fit(X_train, y_train)

    # Evaluate on validation data
    evaluate_model(clf, X_val, y_val)

    joblib.dump(clf, model_path)
    print(f"Model saved to {model_path}")
    return clf

def load_model(model_path=MODEL_PATH):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"No model found at {model_path}")
    return joblib.load(model_path)

