"""Lightweight profiling helpers for CLI entrypoints."""

from __future__ import annotations

import cProfile
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator


@contextmanager
def profile_section(enabled: bool, *, output: Path | None = None) -> Iterator[None]:
    """Optionally profile a block and write stats to ``output``."""

    if not enabled:
        yield
        return

    profiler = cProfile.Profile()
    profiler.enable()
    start = time.perf_counter()
    try:
        yield
    finally:
        profiler.disable()
        if output is not None:
            output.parent.mkdir(parents=True, exist_ok=True)
            profiler.dump_stats(str(output))
        elapsed = time.perf_counter() - start
        _ = elapsed
