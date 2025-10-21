from __future__ import annotations

from typing import Any

import pytest
import requests

from crypto_analyzer.data.sentiment_index_fetcher import fetch_fear_greed_index


class DummyResponse:
    def __init__(self, payload: dict[str, Any]) -> None:
        self._payload = payload

    def json(self) -> dict[str, Any]:
        return self._payload

    def raise_for_status(self) -> None:  # pragma: no cover - mirrors requests
        return None


class DummySession:
    def __init__(self, payload: dict[str, Any]) -> None:
        self._payload = payload
        self.calls: list[dict[str, Any]] = []

    def get(self, url: str, params: dict[str, Any], timeout: int | None = None) -> DummyResponse:
        self.calls.append({"url": url, "params": params, "timeout": timeout})
        return DummyResponse(self._payload)


def test_fetch_fear_greed_index_parses_payload() -> None:
    payload = {
        "data": [
            {
                "value": "40",
                "value_classification": "Fear",
                "timestamp": "1",
                "time_until_update": "3600",
            },
            {
                "value": "60",
                "value_classification": "Greed",
                "timestamp": "2",
            },
        ]
    }
    session = DummySession(payload)

    frame = fetch_fear_greed_index(limit=2, session=session)

    assert list(frame.columns) == ["timestamp", "value", "classification", "time_until_update"]
    assert frame.shape[0] == 2
    assert frame.loc[0, "value"] == pytest.approx(40)
    assert frame.loc[1, "classification"] == "Greed"
    assert session.calls[0]["params"]["limit"] == 2


def test_fetch_fear_greed_index_handles_errors() -> None:
    class FailingSession:
        def get(self, *_: Any, **__: Any) -> Any:
            raise requests.RequestException("oops")

    frame = fetch_fear_greed_index(session=FailingSession())

    assert frame.empty
    assert list(frame.columns) == ["timestamp", "value", "classification", "time_until_update"]


@pytest.mark.parametrize("limit", [0, -1])
def test_fetch_fear_greed_index_validates_limit(limit: int) -> None:
    session = DummySession({"data": []})
    with pytest.raises(ValueError):
        fetch_fear_greed_index(limit=limit, session=session)
