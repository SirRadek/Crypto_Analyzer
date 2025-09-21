"""Utilities for persisting experiment artefacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal, overload

import joblib
import pandas as pd

from .logging import configure_logging, resolve_run_id
from .repro import set_seeds

Location = Literal["outputs", "reports"]

__all__ = [
    "build_path",
    "initialize_run",
    "save_csv",
    "save_json",
    "save_model",
    "save_png",
]


def build_path(
    filename: str,
    *,
    run_id: str,
    location: Location = "outputs",
    base_dir: str | Path | None = None,
    subdir: str | Path | None = None,
    ensure_parents: bool = True,
) -> Path:
    """Return a path under ``outputs/run_id=...`` or ``reports``."""

    if location == "outputs":
        root = Path(base_dir or "outputs") / f"run_id={run_id}"
    elif location == "reports":
        root = Path(base_dir or "reports")
    else:  # pragma: no cover - defensive guard
        raise ValueError(f"Unsupported location: {location}")

    if subdir is not None:
        root = root / Path(subdir)

    if ensure_parents:
        root.mkdir(parents=True, exist_ok=True)

    return root / filename


def initialize_run(
    run_id: str | None = None,
    *,
    outputs_root: str | Path = "outputs",
    reports_root: str | Path = "reports",
    deterministic_torch: bool = False,
) -> tuple[str, Path, Path]:
    """Prepare run/report directories, configure logging and set seeds."""

    resolved_run_id = resolve_run_id(run_id)
    run_dir = Path(outputs_root) / f"run_id={resolved_run_id}"
    reports_dir = Path(reports_root)

    run_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    configure_logging(resolved_run_id, run_dir=run_dir)
    set_seeds(resolved_run_id, deterministic=deterministic_torch)

    return resolved_run_id, run_dir, reports_dir


def save_json(
    data: Any,
    filename: str,
    *,
    run_id: str,
    location: Location = "outputs",
    base_dir: str | Path | None = None,
    subdir: str | Path | None = None,
    indent: int = 2,
) -> Path:
    """Persist ``data`` as JSON under the requested location."""

    target = build_path(
        filename,
        run_id=run_id,
        location=location,
        base_dir=base_dir,
        subdir=subdir,
    )
    with target.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=indent, ensure_ascii=False)
        handle.write("\n")
    return target


def save_csv(
    frame: pd.DataFrame,
    filename: str,
    *,
    run_id: str,
    location: Location = "outputs",
    base_dir: str | Path | None = None,
    subdir: str | Path | None = None,
    index: bool = False,
    float_format: str | None = None,
) -> Path:
    """Persist a :class:`pandas.DataFrame` to CSV."""

    target = build_path(
        filename,
        run_id=run_id,
        location=location,
        base_dir=base_dir,
        subdir=subdir,
    )
    frame.to_csv(target, index=index, float_format=float_format)
    return target


@overload
def save_png(
    image: "pd.Series | pd.DataFrame",
    filename: str,
    *,
    run_id: str,
    location: Location = "outputs",
    base_dir: str | Path | None = None,
    subdir: str | Path | None = None,
    dpi: int = 300,
) -> Path: ...


@overload
def save_png(
    image: "Any",
    filename: str,
    *,
    run_id: str,
    location: Location = "outputs",
    base_dir: str | Path | None = None,
    subdir: str | Path | None = None,
    dpi: int = 300,
) -> Path: ...


def save_png(
    image: Any,
    filename: str,
    *,
    run_id: str,
    location: Location = "outputs",
    base_dir: str | Path | None = None,
    subdir: str | Path | None = None,
    dpi: int = 300,
) -> Path:
    """Save a matplotlib ``Figure`` or array-like as PNG."""

    target = build_path(
        filename,
        run_id=run_id,
        location=location,
        base_dir=base_dir,
        subdir=subdir,
    )

    try:
        from matplotlib.figure import Figure
    except ModuleNotFoundError:  # pragma: no cover - optional dependency
        raise RuntimeError("matplotlib is required to save PNG artefacts") from None

    if isinstance(image, Figure):
        figure = image
    elif hasattr(image, "get_figure"):
        figure = image.get_figure()
        if figure is None:
            raise ValueError("Unable to resolve figure from provided image object")
    else:  # pragma: no cover - defensive branch
        raise TypeError("save_png expects a matplotlib Figure or Axes")

    figure.savefig(target, dpi=dpi, bbox_inches="tight")
    return target


def save_model(
    model: Any,
    filename: str = "model.joblib",
    *,
    run_id: str,
    base_dir: str | Path | None = None,
    subdir: str | Path | None = None,
) -> Path:
    """Persist ``model`` using :mod:`joblib` under the run directory."""

    target = build_path(
        filename,
        run_id=run_id,
        location="outputs",
        base_dir=base_dir,
        subdir=subdir,
    )
    joblib.dump(model, target)
    return target

