import json
import time

from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier

from crypto_analyzer.eval.oob import fit_incremental_forest, halving_random_search


def test_incremental_forest_faster_and_oob(tmp_path):
    X, y = make_classification(n_samples=1000, n_features=20, random_state=0)

    start = time.perf_counter()
    baseline = RandomForestClassifier(
        n_estimators=400, oob_score=True, bootstrap=True, n_jobs=-1, random_state=0
    )
    baseline.fit(X, y)
    base_time = time.perf_counter() - start
    base_oob = baseline.oob_score_

    log_path = tmp_path / "run.json"
    start = time.perf_counter()
    model, oob_scores = fit_incremental_forest(
        X,
        y,
        RandomForestClassifier,
        step=100,
        max_estimators=400,
        tol=1e-3,
        random_state=0,
        log_path=str(log_path),
    )
    inc_time = time.perf_counter() - start
    inc_oob = oob_scores[-1]

    assert inc_time < base_time
    assert inc_oob >= base_oob - 0.01
    data = json.loads(log_path.read_text())
    assert data["oob_scores"] == oob_scores
    assert data["params"]["n_estimators"] == model.n_estimators


def test_halving_random_search_returns_params():
    X, y = make_classification(n_samples=500, n_features=10, random_state=0)
    params = halving_random_search(X, y, RandomForestClassifier, random_state=0)
    keys = {
        "n_estimators",
        "max_depth",
        "max_features",
        "min_samples_split",
        "min_samples_leaf",
    }
    assert keys.issubset(params.keys())
