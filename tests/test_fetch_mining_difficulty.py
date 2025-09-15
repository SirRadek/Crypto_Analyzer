import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))
from api import backfill_onchain_history as mod


class DummyResp:
    def __init__(self, data):
        self._data = data

    def raise_for_status(self):
        return None

    def json(self):
        return self._data


class DummySession:
    def __init__(self, data):
        self.data = data

    def get(self, url, timeout=15):
        return DummyResp(self.data)


def test_fetch_mining_difficulty_list():
    start = pd.Timestamp("2021-01-01 00:00", tz="UTC")
    end = start + pd.Timedelta(hours=1)
    sample = [
        [start.timestamp(), 100000, 1.0, 1.05],
        [(start + pd.Timedelta(minutes=30)).timestamp(), 100050, 1.0, 0.95],
    ]
    session = DummySession(sample)
    df = mod.fetch_mining_difficulty(start, end, session)
    assert len(df) == 2
    assert df.iloc[0]["onch_height"] == 100000
    change_second = (0.95 - 1.0) * 100.0
    assert df.iloc[1]["onch_diff_change_pct"] == change_second
