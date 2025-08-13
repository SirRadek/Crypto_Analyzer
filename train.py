import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

MODEL_PATH = "ml/model.pkl"

def train_model(X, y, model_path=MODEL_PATH):
    """
    Trains a RandomForestClassifier and saves it to disk.
    X: features DataFrame
    y: target Series (e.g., 1 = price up, 0 = price down)
    """
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X, y)
    joblib.dump(clf, model_path)
    print(f"Model saved to {model_path}")
    return clf

def load_model(model_path=MODEL_PATH):
    """
    Loads a trained model from disk.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"No model found at {model_path}")
    return joblib.load(model_path)
