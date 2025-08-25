import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

MODEL_PATH = "ml/model.pkl"

def train_model(X, y, model_path=MODEL_PATH):
    clf = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        n_jobs=-1,
        min_samples_leaf=2,
        verbose=1  # <- prints training progress
    )
    clf.fit(X, y)
    joblib.dump(clf, model_path)
    print(f"Model saved to {model_path}")
    return clf

def load_model(model_path=MODEL_PATH):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"No model found at {model_path}")
    return joblib.load(model_path)

