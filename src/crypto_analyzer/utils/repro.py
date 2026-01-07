"""Reproducibility helpers."""

from __future__ import annotations

import contextlib
import hashlib
import os
import random

import numpy as np

__all__ = ["set_seeds"]


def _seed_from_run_id(run_id: str) -> int:
    digest = hashlib.sha256(run_id.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big")


def set_seeds(run_id: str, *, deterministic: bool = False) -> int:
    """Seed random number generators using a stable hash of ``run_id``."""

    seed = _seed_from_run_id(run_id)
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))

    try:
        import torch
    except ModuleNotFoundError:  # pragma: no cover - optional dependency
        torch = None  # type: ignore[assignment]

    if torch is not None:
        torch.manual_seed(seed)
        if deterministic:
            with contextlib.suppress(
                RuntimeError, AttributeError
            ):  # pragma: no cover - env dependent
                torch.use_deterministic_algorithms(True)

    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    return seed
