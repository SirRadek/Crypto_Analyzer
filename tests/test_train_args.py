import importlib

import pytest


@pytest.mark.parametrize(
    "module_name",
    [
        "ml.train_classifier",
        "ml.train_meta_classifier",
    ],
)
def test_parses_args(module_name):
    mod = importlib.import_module(module_name)
    args = mod.parse_args(
        [
            "--train-start",
            "2024-01-01",
            "--train-end",
            "2024-01-31",
            "--horizon",
            "15m",
            "--step",
            "1m",
            "--eval-frac",
            "0.2",
        ]
    )
    assert args.train_start == "2024-01-01"
    args2 = mod.parse_args(
        [
            "--train-window",
            "10 days",
            "--horizon",
            "1h",
            "--step",
            "30m",
            "--eval-split",
            "2024-02-01:2024-02-10",
        ]
    )
    assert args2.train_window == "10 days"
