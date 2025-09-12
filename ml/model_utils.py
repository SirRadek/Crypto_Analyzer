import os

import joblib
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)


def save_model(model, path):
    """
    Saves a model to disk using joblib.
    """
    joblib.dump(model, path)
    print(f"Model saved to {path}")


def load_model(path):
    """
    Loads a model from disk.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found at {path}")
    return joblib.load(path)


def match_model_features(df, model):
    """Align ``df`` columns with features expected by ``model``.

    Extra columns are dropped and missing ones are filled with zeros so that
    the returned dataframe has the same column order as used during model
    training. If the model does not expose ``feature_names_in_`` the original
    ``df`` is returned unchanged.
    """

    feature_names = getattr(model, "feature_names_in_", None)
    if feature_names is None:
        return df
    return df.reindex(columns=feature_names, fill_value=0)


def evaluate_model(model, X_test, y_test):
    """Print evaluation metrics for classification models.

    Returns
    -------
    tuple
        (accuracy, f1) scores for further analysis.
    """
    preds = model.predict(X_test)
    if hasattr(preds, "to_numpy"):
        preds = preds.to_numpy()  # type: ignore[assignment]
    if hasattr(y_test, "to_numpy"):
        y_test = y_test.to_numpy()  # type: ignore[assignment]
    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds, average="weighted")
    print("Accuracy:", acc)
    print("F1 score:", f1)
    print("Classification report:\n", classification_report(y_test, preds))
    print("Confusion matrix:\n", confusion_matrix(y_test, preds))
    return acc, f1
