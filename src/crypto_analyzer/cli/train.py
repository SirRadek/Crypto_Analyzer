"""CLI entrypoint for the training pipeline."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from typing import Any

from crypto_analyzer.pipeline import run_pipeline
from crypto_analyzer.utils.config import CONFIG
from crypto_analyzer.utils.helpers import get_logger, set_cpu_limit
from crypto_analyzer.utils.progress import timed

logger = get_logger(__name__)

CPU_LIMIT = CONFIG.cpu_limit

__all__ = ["main"]


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the classification pipeline")
    parser.add_argument("--task", default="clf")
    parser.add_argument("--horizon", type=int, default=120)
    parser.add_argument("--use_onchain", action="store_true")
    parser.add_argument("--txn_cost_bps", type=float, default=1.0)
    parser.add_argument("--split_params", type=str, default="{}")
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--out_dir", type=str, default="outputs")
    parser.add_argument("--cpu_limit", type=int, default=CPU_LIMIT)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.cpu_limit:
        set_cpu_limit(args.cpu_limit)

    try:
        split_params: dict[str, Any] = json.loads(args.split_params)
    except json.JSONDecodeError as exc:  # pragma: no cover - defensive
        raise SystemExit(f"Invalid --split_params JSON: {exc}") from exc

    with timed("train_pipeline"):
        run_pipeline(
            task=args.task,
            horizon=args.horizon,
            use_onchain=True if args.use_onchain else None,
            txn_cost_bps=args.txn_cost_bps,
            split_params=split_params,
            gpu=args.gpu,
            out_dir=args.out_dir,
            config=CONFIG,
        )
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
