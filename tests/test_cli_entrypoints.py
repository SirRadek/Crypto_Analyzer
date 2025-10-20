"""Smoke tests covering the Typer CLI entrypoints."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

typer = pytest.importorskip("typer")
from typer.testing import CliRunner


RUNNER = CliRunner()
SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"


def _load_app(script: str):
    module_path = SCRIPTS_DIR / f"{script}.py"
    spec = importlib.util.spec_from_file_location(f"cli_{script}", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)  # type: ignore[call-arg]
    return getattr(module, "app")


def test_make_features_help():
    app = _load_app("make_features")
    result = RUNNER.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "--store" in result.stdout


def test_train_help():
    app = _load_app("train")
    result = RUNNER.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "--store" in result.stdout


def test_predict_help():
    app = _load_app("predict")
    result = RUNNER.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "--threshold" in result.stdout
