import numpy as np
import pandas as pd

from crypto_analyzer.models.stacking import (
    StackingConfig,
    train_stacking_meta_learner,
    stack_predict,
)


def test_train_stacking_meta_learner_and_predict():
    base_preds = pd.DataFrame(
        {
            "model_a": [0.2, 0.8, 0.4, 0.7, 0.9, 0.1],
            "model_b": [0.3, 0.6, 0.5, 0.8, 0.85, 0.2],
        }
    )
    target = np.array([0, 1, 0, 1, 1, 0])

    result = train_stacking_meta_learner(
        base_preds,
        target,
        config=StackingConfig(cv_splits=3, calibrate=True),
    )
    assert set(result.feature_names) == {"model_a", "model_b"}
    assert "log_loss" in result.metrics

    probs = stack_predict(result.model, base_preds, proba=True)
    assert probs.shape == (len(base_preds),)
    labels = stack_predict(result.model, base_preds, proba=False)
    assert set(np.unique(labels)).issubset({0, 1})

