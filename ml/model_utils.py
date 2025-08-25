import joblib
import os
from sklearn.metrics import accuracy_score, classification_report

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
    """
    Prints basic evaluation metrics for classification.
    """
    preds = model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, preds))
    print("Classification report:\n", classification_report(y_test, preds))
