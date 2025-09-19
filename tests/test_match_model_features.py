import importlib.util
import pathlib
import pandas as pd

# Load model utilities without importing the full package to avoid heavy optional
# dependencies such as xgboost during import.
module_path = (
    pathlib.Path(__file__).resolve().parents[1]
    / "src"
    / "crypto_analyzer"
    / "models"
    / "utils.py"
)
spec = importlib.util.spec_from_file_location("model_utils", module_path)
model_utils = importlib.util.module_from_spec(spec)
spec.loader.exec_module(model_utils)
match_model_features = model_utils.match_model_features


class DummyModel:
    """Model exposing duplicate feature names via scikit-learn API."""

    feature_names_in_ = ["a", "b", "a"]


def test_duplicates_are_deduped():
    df = pd.DataFrame({"a": [1.0], "b": [2.0]})
    aligned = match_model_features(df, DummyModel())
    assert list(aligned.columns) == ["a", "b"]
