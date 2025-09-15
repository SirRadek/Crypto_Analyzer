import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))
from api import backfill_onchain_history as mod


def test_fetch_mining_difficulty_list(monkeypatch):
    start = pd.Timestamp('2021-01-01 00:00', tz='UTC')
    end = start + pd.Timedelta(hours=1)
    sample = [
        [start.timestamp(), 100000, 1.0, 1.05],
        [(start + pd.Timedelta(minutes=30)).timestamp(), 100050, 1.0, 0.95],
    ]
    monkeypatch.setattr(mod, '_get_json', lambda sess, url, timeout=30: sample)
    df = mod.fetch_mining_difficulty(start, end, None)
    assert len(df) == 2
    progress_first = (100000 % 2016) / 2016 * 100
    remaining_first = 2016 - (100000 % 2016)
    change_second = (0.95 - 1) * 100
    assert df.iloc[0]['onch_diff_progress_pct'] == progress_first
    assert df.iloc[1]['onch_diff_change_pct'] == change_second
    expected_retarget = int((start + pd.Timedelta(seconds=remaining_first * 600)).timestamp() * 1000)
    assert df.iloc[0]['onch_retarget_ts'] == expected_retarget
