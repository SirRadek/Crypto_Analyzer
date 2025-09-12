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
    with open(config_path) as f:
        return yaml.safe_load(f)


def timestamp_to_datetime(ts):
    """
    Converts UNIX timestamp (ms) to Python datetime.
    """
    from datetime import datetime

    return datetime.fromtimestamp(ts / 1000.0)


def set_cpu_limit(cores: int) -> 20:
    """Limit the process to ``cores`` CPU cores if possible.

    If ``psutil`` or CPU affinity is not available, the function silently
    falls back without raising an error.
    """
    try:
        import psutil

        cores = max(1, int(cores))
        proc = psutil.Process()
        proc.cpu_affinity(list(range(cores)))
    except Exception:
        pass
