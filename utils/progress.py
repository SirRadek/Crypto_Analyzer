import time
from contextlib import contextmanager

from .helpers import get_logger

logger = get_logger(__name__)


def p(msg: str) -> None:
    """Log *msg* using the project logger."""
    logger.info(msg)


def step(i: int, n: int, msg: str) -> None:
    """Log progress of step *i* out of *n* with message *msg*."""
    p(f"[{i}/{n}] {msg}")


@contextmanager
def timed(label: str):
    """Context manager logging the runtime of a labeled block."""

    t0 = time.perf_counter()
    p(f"{label} ...")
    try:
        yield
    finally:
        dt = time.perf_counter() - t0
        p(f"{label} done in {dt:.2f}s")
