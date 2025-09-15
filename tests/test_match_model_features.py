import importlib.util
import pathlib
import pandas as pd

# Load ml/model_utils.py without importing the ml package, which would require
# heavy optional dependencies such as xgboost.
module_path = pathlib.Path(__file__).resolve().parents[1] / "ml" / "model_utils.py"
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
