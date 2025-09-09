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

def evaluate_model(model, X_test, y_test):
    """Print evaluation metrics for classification models.

    Returns
    -------
    tuple
        (accuracy, f1) scores for further analysis.
    """
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds, average="weighted")
    print("Accuracy:", acc)
    print("F1 score:", f1)
    print("Classification report:\n", classification_report(y_test, preds))
    print("Confusion matrix:\n", confusion_matrix(y_test, preds))
    return acc, f1
