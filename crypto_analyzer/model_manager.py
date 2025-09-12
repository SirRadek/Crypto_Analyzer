from __future__ import annotations

import argparse
import os
from collections.abc import Sequence
from pathlib import Path
from tempfile import NamedTemporaryFile

PROJECT_ROOT = Path(
    os.environ.get("CRYPTO_ANALYZER_PROJECT_ROOT", Path(__file__).resolve().parents[1])
)
MODELS_ROOT = PROJECT_ROOT / "models"


def _ensure_models_dir(dir: Path) -> Path:
    resolved = dir.resolve()
    models_dir = MODELS_ROOT.resolve()
    if not resolved.is_dir():
        raise ValueError(f"{dir} is not a directory")
    if not resolved.is_relative_to(models_dir):
        raise ValueError("dir must be inside project_root/models")
    return resolved


def list_artifacts(
    stem: str, dir: Path, patterns: Sequence[str] = ("*.joblib", "*.pkl", "*.npy", "*.bin")
) -> list[Path]:
    """List model artifacts matching stem within a directory.

    Parameters
    ----------
    stem:
        Prefix of the file name without extension.
    dir:
        Directory to search. Must reside under ``PROJECT_ROOT/models``.
    patterns:
        Glob patterns of file types to consider.
    """
    base = _ensure_models_dir(dir)
    files: list[Path] = []
    for pat in patterns:
        files.extend(p for p in base.glob(pat) if p.stem.startswith(stem))
    return sorted(files)


def purge_artifacts(
    stem: str,
    dir: Path,
    patterns: Sequence[str] = ("*.joblib", "*.pkl", "*.npy", "*.bin"),
    confirm: bool = False,
) -> list[Path]:
    """Delete model artifacts if ``confirm`` is True.

    Returns the list of matched files regardless of deletion."""
    files = list_artifacts(stem, dir, patterns)
    if confirm:
        for p in files:
            try:
                p.unlink()
            except FileNotFoundError:
                pass
    return files


def atomic_write(path: Path, data: bytes) -> None:
    """Atomically write ``data`` to ``path``.

    Uses :func:`os.replace` to guarantee atomicity within the same filesystem."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(delete=False, dir=str(path.parent)) as tmp:
        tmp.write(data)
        tmp.flush()
        os.fsync(tmp.fileno())
        tmp_path = Path(tmp.name)
    os.replace(tmp_path, path)


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Manage model artifacts")
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--list", action="store_true", help="List artifacts")
    action.add_argument("--purge", action="store_true", help="Purge artifacts")
    parser.add_argument("--stem", required=True, help="File name stem to match")
    parser.add_argument("--dir", type=Path, required=True, help="Directory under models/")
    confirm = parser.add_mutually_exclusive_group()
    confirm.add_argument("--dry-run", action="store_true", help="Do not delete anything")
    confirm.add_argument("--confirm", action="store_true", help="Actually delete files")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.list:
        files = list_artifacts(args.stem, args.dir)
        for f in files:
            print(f)
        return 0

    # Purge mode
    files = purge_artifacts(args.stem, args.dir, confirm=args.confirm)
    for f in files:
        print(f)
    if args.dry_run or not args.confirm:
        print("Dry run - no files deleted")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
