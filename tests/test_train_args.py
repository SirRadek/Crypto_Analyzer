import importlib

import pytest


@pytest.mark.parametrize(
    "module_name",
    [
        "crypto_analyzer.models.train_classifier",
        "crypto_analyzer.models.train_meta_classifier",
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
            "1d",
            "--step",
            "1d",
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
            "7d",
            "--step",
            "1d",
            "--eval-split",
            "2024-02-01:2024-02-10",
        ]
    )
    assert args2.train_window == "10 days"


def test_train_cli_conformal_parse():
    pytest.importorskip("xgboost")
    from crypto_analyzer.models import train as train_module

    args = train_module.parse_args(
        [
            "--task",
            "clf",
            "--horizon",
            "1440",
            "--conformal",
            "alpha=0.05",
        ]
    )
    assert args.conformal == {"alpha": 0.05}

    args_default = train_module.parse_args([
        "--task",
        "clf",
        "--horizon",
        "1440",
        "--conformal",
    ])
    assert args_default.conformal == {"alpha": 0.1}
