"""Feature cache helpers for CLI entrypoints."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from crypto_analyzer.utils.config import CONFIG, FeatureSettings

__all__ = ["build_cache_key", "build_cache_payload", "cache_paths", "default_cache_dir"]


def default_cache_dir() -> Path:
    """Resolve the default feature cache directory."""

    return CONFIG.database.feature_store or (CONFIG.runtime.cache_dir / "features")


def _settings_payload(settings: FeatureSettings) -> dict[str, Any]:
    if hasattr(settings, "model_dump"):
        return settings.model_dump(mode="json")
    return {
        "include_onchain": settings.include_onchain,
        "include_orderbook": settings.include_orderbook,
        "include_derivatives": settings.include_derivatives,
        "include_sentiment": settings.include_sentiment,
        "forward_fill_limit": settings.forward_fill_limit,
        "fillna_value": settings.fillna_value,
    }


def build_cache_key(payload: dict[str, Any]) -> str:
    """Hash the cache payload deterministically."""

    raw = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def cache_paths(cache_dir: Path, cache_key: str) -> tuple[Path, Path]:
    """Return parquet and metadata paths for the cache key."""

    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f"{cache_key}.parquet", cache_dir / f"{cache_key}.json"


def build_cache_payload(
    *,
    source: str,
    symbol: str,
    input_path: Path | None,
    settings: FeatureSettings,
    store_label: str | None,
    store_location: str | None,
    latest_open_time: int | None,
) -> dict[str, Any]:
    """Build a stable cache payload for feature outputs."""

    payload: dict[str, Any] = {
        "source": source,
        "symbol": symbol,
        "settings": _settings_payload(settings),
        "store_label": store_label,
        "store_location": store_location,
        "latest_open_time": latest_open_time,
    }
    if input_path is not None and input_path.exists():
        stat = input_path.stat()
        payload["input_path"] = str(input_path)
        payload["input_mtime"] = int(stat.st_mtime)
        payload["input_size"] = int(stat.st_size)
    return payload
