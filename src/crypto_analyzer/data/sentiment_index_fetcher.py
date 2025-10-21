"""Retrieve Crypto Fear & Greed index values from alternative.me."""

from __future__ import annotations

from typing import Any

import pandas as pd
import requests

from crypto_analyzer.utils.logging import get_logger

LOGGER = get_logger(__name__)

_FEAR_GREED_ENDPOINT = "https://api.alternative.me/fng/"


def fetch_fear_greed_index(
    *, limit: int = 1, session: requests.Session | None = None
) -> pd.DataFrame:
    """Fetch historical Fear & Greed index readings."""

    if limit <= 0:
        raise ValueError("limit must be a positive integer")

    params = {"limit": limit, "format": "json"}
    sess = session or requests.Session()
    try:
        response = sess.get(_FEAR_GREED_ENDPOINT, params=params, timeout=10)
        response.raise_for_status()
        payload: dict[str, Any] = response.json() or {}
    except (requests.RequestException, ValueError) as exc:
        LOGGER.warning("Failed to fetch Fear & Greed index", exc_info=exc)
        return pd.DataFrame(columns=["timestamp", "value", "classification", "time_until_update"])

    data = payload.get("data", [])
    frame = pd.DataFrame(data)
    if frame.empty:
        return pd.DataFrame(columns=["timestamp", "value", "classification", "time_until_update"])

    frame["timestamp"] = pd.to_datetime(frame["timestamp"], unit="s", utc=True, errors="coerce")
    frame["value"] = pd.to_numeric(frame["value"], errors="coerce")
    frame.rename(columns={"value_classification": "classification"}, inplace=True)
    if "time_until_update" in frame:
        frame["time_until_update"] = pd.to_numeric(frame["time_until_update"], errors="coerce")
    desired_columns = ["timestamp", "value", "classification", "time_until_update"]
    for column in desired_columns:
        if column not in frame.columns:
            frame[column] = pd.NA
    frame = frame[desired_columns]
    frame = frame.sort_values("timestamp").reset_index(drop=True)
    return frame


__all__ = ["fetch_fear_greed_index"]
