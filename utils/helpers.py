import os
import yaml

def ensure_dir_exists(path):
    """
    Creates a directory if it does not exist.
    """
    if not os.path.exists(path):
        os.makedirs(path)

def load_yaml_config(config_path):
    """
    Loads configuration from a YAML file.
    """
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def timestamp_to_datetime(ts):
    """
    Converts UNIX timestamp (ms) to Python datetime.
    """
    from datetime import datetime
    return datetime.fromtimestamp(ts / 1000.0)
