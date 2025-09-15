import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))
from api import backfill_onchain_history as mod


class _DummyResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


class _DummySession:
    def __init__(self, payload):
        self._payload = payload
        self.calls: list[tuple[str, int]] = []

    def get(self, url, timeout=0):
        self.calls.append((url, timeout))
        return _DummyResponse(self._payload)


def test_fetch_hashrate_nested_lists():
    start = pd.Timestamp("2024-01-01", tz="UTC")
    end = pd.Timestamp("2024-01-02", tz="UTC")
    payload = {
        "hashrate": {
            "data": [
                [start.timestamp(), "101.5"],
                [end.timestamp(), 98.0],
                [end.timestamp() + 86400, 5.0],
            ]
        }
    }
    session = _DummySession(payload)
    df = mod.fetch_hashrate(start, end, session)
    assert not df.empty
    assert list(df.index) == [start, end]
    assert list(df["onch_hashrate_ehs"]) == [101.5, 98.0]


def test_fetch_hashrate_dict_series():
    start = pd.Timestamp("2024-03-01", tz="UTC")
    end = pd.Timestamp("2024-03-01 02:00", tz="UTC")
    payload = {
        "hashrate": {
            "series": [
                {"time": "2024-03-01T00:00:00Z", "hashrate": "75.1"},
                {"time": "2024-03-01T01:00:00Z", "hashrate": "77.3"},
                {"time": "2024-03-01T03:00:00Z", "hashrate": "999"},
            ]
        }
    }
    session = _DummySession(payload)
    df = mod.fetch_hashrate(start, end, session)
    assert list(df.index) == [pd.Timestamp("2024-03-01T00:00Z"), pd.Timestamp("2024-03-01T01:00Z")]
    assert df["onch_hashrate_ehs"].tolist() == [75.1, 77.3]


def test_fetch_hashrate_invalid_payload():
    start = pd.Timestamp("2024-01-01", tz="UTC")
    end = pd.Timestamp("2024-01-02", tz="UTC")
    payload = {"status": "ok", "data": ["n/a", None]}
    session = _DummySession(payload)
    df = mod.fetch_hashrate(start, end, session)
    assert df.empty
    assert list(df.columns) == ["onch_hashrate_ehs"]
