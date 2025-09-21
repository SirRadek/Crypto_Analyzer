"""Project logging helpers with JSON formatting and run context support."""

from __future__ import annotations

import json
import logging
from contextvars import ContextVar
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional

__all__ = [
    "JsonLogFormatter",
    "configure_logging",
    "current_run_id",
    "generate_run_id",
    "get_logger",
    "resolve_run_id",
]


_ROOT_LOGGER_NAME = "crypto_analyzer"
_RUN_ID: ContextVar[Optional[str]] = ContextVar("crypto_run_id", default=None)
_RUN_DIR: ContextVar[Optional[Path]] = ContextVar("crypto_run_dir", default=None)
_CONFIGURED: ContextVar[bool] = ContextVar("crypto_log_configured", default=False)


def generate_run_id() -> str:
    """Return a timestamp-based identifier used to track experiment runs."""

    return datetime.utcnow().strftime("%Y%m%d_%H%M%S")


def resolve_run_id(run_id: str | None = None) -> str:
    """Normalise ``run_id`` ensuring a valid identifier is returned."""

    return run_id or generate_run_id()


class JsonLogFormatter(logging.Formatter):
    """Emit log records as structured JSON objects."""

    def format(self, record: logging.LogRecord) -> str:  # noqa: D401 - inherited doc
        payload: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created, tz=timezone.utc)
            .isoformat()
            .replace("+00:00", "Z"),
            "level": record.levelname,
            "name": record.name,
            "message": record.getMessage(),
        }

        run_id = _RUN_ID.get()
        if run_id:
            payload["run_id"] = run_id

        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        if record.stack_info:
            payload["stack"] = record.stack_info

        for key in ("event", "extra"):
            if key in record.__dict__:
                payload[key] = record.__dict__[key]

        return json.dumps(payload, ensure_ascii=False)


def _ensure_root_logger(level: int = logging.INFO) -> logging.Logger:
    configured = _CONFIGURED.get()
    root_logger = logging.getLogger(_ROOT_LOGGER_NAME)
    if configured:
        return root_logger

    root_logger.setLevel(level)
    root_logger.propagate = False

    handler = logging.StreamHandler()
    handler.setFormatter(JsonLogFormatter())
    root_logger.addHandler(handler)

    _CONFIGURED.set(True)
    return root_logger


def _attach_file_handler(logger: logging.Logger, run_dir: Path) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "run.log"
    for handler in logger.handlers:
        if isinstance(handler, logging.FileHandler) and Path(getattr(handler, "baseFilename", "")) == log_path:
            return

    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(JsonLogFormatter())
    logger.addHandler(file_handler)


def configure_logging(
    run_id: str,
    *,
    run_dir: Path | None = None,
    level: int = logging.INFO,
) -> logging.Logger:
    """Configure project logging for the provided ``run_id``."""

    _RUN_ID.set(run_id)
    logger = _ensure_root_logger(level=level)

    if run_dir is not None:
        _RUN_DIR.set(run_dir)
        _attach_file_handler(logger, run_dir)

    return logger


def get_logger(name: str | None = None) -> logging.Logger:
    """Return a child logger scoped to the Crypto Analyzer namespace."""

    base = _ROOT_LOGGER_NAME
    if name and not name.startswith(base):
        full_name = f"{base}.{name}"
    else:
        full_name = name or base

    logger = logging.getLogger(full_name)
    if not _CONFIGURED.get():
        _ensure_root_logger()
    return logger


def current_run_id() -> str | None:
    """Return the run identifier associated with the active logging context."""

    return _RUN_ID.get()


def current_run_dir() -> Path | None:
    """Return the active run directory if available."""

    return _RUN_DIR.get()


def add_extra_handlers(handlers: Iterable[logging.Handler]) -> None:
    """Attach additional handlers to the project root logger."""

    logger = _ensure_root_logger()
    for handler in handlers:
        logger.addHandler(handler)

