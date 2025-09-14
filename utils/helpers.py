from __future__ import annotations

import logging
import os
from datetime import datetime
from typing import Any

import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)


def get_logger(name: str | None = None) -> logging.Logger:
    """Return a module-level logger configured for the project."""
    return logging.getLogger(name if name else __name__)


def ensure_dir_exists(path: str | os.PathLike[str]) -> None:
    """Create directory *path* if it does not exist."""
    if not os.path.exists(path):
        os.makedirs(path)


def load_yaml_config(config_path: str) -> dict[str, Any]:
    """Load configuration from a YAML file.

    Parameters
    ----------
    config_path:
        Path to YAML file.

    Returns
    -------
    dict[str, Any]
        Parsed configuration.
    """

    try:
        with open(config_path, encoding="utf-8") as f:
            return yaml.safe_load(f)
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Config file not found: {config_path}") from exc


def timestamp_to_datetime(ts: int) -> datetime:
    """Convert UNIX timestamp (ms) to :class:`datetime.datetime`."""
    return datetime.fromtimestamp(ts / 1000.0)


def set_cpu_limit(cores: int) -> None:
    """Limit process to ``cores`` CPU cores if possible.

    Parameters
    ----------
    cores:
        Number of CPU cores to allow.

    Notes
    -----
    Falls back silently if ``psutil`` or CPU affinity is unavailable.
    """

    try:
        import psutil

        cores = max(1, int(cores))
        proc = psutil.Process()
        proc.cpu_affinity(list(range(cores)))
    except Exception:
        pass
