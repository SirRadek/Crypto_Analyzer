import time
from contextlib import contextmanager

from tqdm import tqdm


def p(msg: str):
    print(msg, flush=True)


def step(i: int, n: int, msg: str):
    p(f"[{i}/{n}] {msg}")


@contextmanager
def timed(label: str):
    t0 = time.perf_counter()
    p(f"{label} ...")
    try:
        yield
    finally:
        dt = time.perf_counter() - t0
        p(f"{label} done in {dt:.2f}s")


def bar(iterable, desc: str, unit: str = "it"):
    return tqdm(iterable, desc=desc, unit=unit)
