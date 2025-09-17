from __future__ import annotations

from contextlib import suppress
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)


def get_logger(name: str | None = None) -> logging.Logger:
    """Return a module-level logger configured for the project."""
    return logging.getLogger(name if name else __name__)


def ensure_dir_exists(path: str | Path) -> Path:
    """Create *path* (and parents) when it is missing.

    Returning the :class:`pathlib.Path` instance makes it easy to reuse the
    resolved path by callers without converting it repeatedly.
    """

    resolved = Path(path)
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


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

    with suppress(Exception):
        import psutil

        cores = max(1, int(cores))
        proc = psutil.Process()
        proc.cpu_affinity(list(range(cores)))
