from __future__ import annotations

import logging
import os

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
