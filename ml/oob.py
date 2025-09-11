import json
from typing import Any

from sklearn.experimental import enable_halving_search_cv  # noqa: F401
from sklearn.model_selection import HalvingRandomSearchCV  # type: ignore[attr-defined]


def halving_random_search(
    X,
    y,
    estimator_class: type[Any],
    random_state: int = 42,
) -> dict[str, Any]:
    """Return best hyperparameters via a small ``HalvingRandomSearchCV``.

    The search spans ``n_estimators``, ``max_depth``, ``max_features``,
    ``min_samples_split`` and ``min_samples_leaf``.
    """

    param_dist: dict[str, list[Any]] = {
        "max_depth": [None, 10, 20, 30],
        "max_features": ["auto", "sqrt", 0.5],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
    }

    base = estimator_class(
        oob_score=True,
        warm_start=True,
        n_jobs=-1,
        random_state=random_state,
    )
    search = HalvingRandomSearchCV(
        base,
        param_dist,
        resource="n_estimators",
        max_resources=400,
        min_resources=50,
        factor=2,
        random_state=random_state,
        n_jobs=-1,
    )
    search.fit(X, y)
    best = dict(search.best_params_)
    best["n_estimators"] = search.best_estimator_.n_estimators
    return best


def fit_incremental_forest(
    X,
    y,
    estimator_class: type[Any],
    *,
    step: int = 50,
    max_estimators: int = 400,
    tol: float = 1e-3,
    random_state: int = 42,
    log_path: str | None = None,
    **params: Any,
) -> tuple[Any, list[float]]:
    """Fit a forest, growing trees until OOB improvement drops below ``tol``.

    Parameters
    ----------
    X, y : array-like
        Training data.
    estimator_class : type
        ``RandomForestClassifier`` or ``RandomForestRegressor``.
    step : int
        Number of trees added per iteration.
    max_estimators : int
        Hard ceiling for trees regardless of OOB improvement.
    tol : float
        Minimum required OOB improvement to continue growing.
    log_path : str, optional
        If provided, JSON is written with final parameters and OOB curve.
    **params : dict
        Extra parameters passed to the estimator constructor.
    """

    model = estimator_class(
        n_estimators=0,
        warm_start=True,
        oob_score=True,
        n_jobs=-1,
        random_state=random_state,
        **params,
    )
    oob_scores: list[float] = []
    while model.n_estimators < max_estimators:
        model.n_estimators += step
        model.fit(X, y)
        oob_scores.append(float(model.oob_score_))
        if len(oob_scores) > 1 and oob_scores[-1] - oob_scores[-2] < tol:
            break

    if log_path:
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump({"params": model.get_params(), "oob_scores": oob_scores}, f)

    return model, oob_scores
