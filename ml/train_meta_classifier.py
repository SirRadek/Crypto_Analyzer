from __future__ import annotations

import argparse
import json
import traceback
from datetime import datetime
from typing import Iterable

from crypto_analyzer.model_manager import MODELS_ROOT, PROJECT_ROOT, atomic_write


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train meta-classifier")
    period = parser.add_mutually_exclusive_group()
    period.add_argument("--train-window")
    period.add_argument("--train-start")
    parser.add_argument("--train-end")
    parser.add_argument("--horizon")
    parser.add_argument("--step")
    eval_group = parser.add_mutually_exclusive_group()
    eval_group.add_argument("--eval-split")
    eval_group.add_argument("--eval-frac", type=float)
    parser.add_argument("--reset-metadata", action="store_true")
    return parser


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.train_start and not args.train_end:
        parser.error("--train-start requires --train-end")
    if args.train_end and not args.train_start:
        parser.error("--train-end requires --train-start")
    if args.train_window and (args.train_start or args.train_end):
        parser.error("--train-window is mutually exclusive with --train-start/--train-end")
    return args


def _reset_metadata() -> None:
    MODELS_ROOT.mkdir(parents=True, exist_ok=True)
    for name in ["model_usage.json", "model_performance.json"]:
        path = MODELS_ROOT / name
        atomic_write(path, b"{}")
        json.loads(path.read_text())


def _log_run(config: dict[str, object]) -> None:
    runs_dir = PROJECT_ROOT / "logs" / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    data = json.dumps({"config": config}, indent=2).encode("utf-8")
    atomic_write(runs_dir / f"run_{ts}.json", data)


def _log_failure(stem: str, exc: BaseException) -> None:
    fails = PROJECT_ROOT / "logs" / "failures.jsonl"
    fails.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "stem": stem,
        "exc_type": type(exc).__name__,
        "message": str(exc),
        "hash": hash(traceback.format_exc()),
    }
    with fails.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    if args.reset_metadata:
        _reset_metadata()
    try:
        _log_run(vars(args))
    except Exception as exc:  # pragma: no cover
        _log_failure("meta_classifier", exc)
        raise
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
