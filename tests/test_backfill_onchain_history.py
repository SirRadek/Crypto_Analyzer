import sqlite3
import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))
from api import backfill_onchain_history as mod


def _make_df(idx: pd.DatetimeIndex) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    fee = pd.DataFrame(
        {
            "onch_fee_wavg_satvb": [1.0, 2.0, 3.0],
            "onch_fee_p50_satvb": [1.0, 2.0, 3.0],
            "onch_fee_p90_satvb": [2.0, 3.0, 4.0],
            "onch_mempool_vsize_vB": [100.0, 200.0, 300.0],
        },
        index=idx,
    )
    bc = pd.DataFrame(
        {
            "onch_fee_fast_satvb": [10.0, 20.0, 30.0],
            "onch_fee_30m_satvb": [11.0, 21.0, 31.0],
            "onch_fee_60m_satvb": [12.0, 22.0, 32.0],
            "onch_fee_min_satvb": [9.0, 19.0, 29.0],
            "onch_mempool_count": [1, 2, 3],
            "onch_mempool_total_fee_sat": [1000, 2000, 3000],
        },
        index=idx,
    )
    diff = pd.DataFrame(
        {
            "onch_difficulty": [1.0, 2.0, 3.0],
            "onch_height": [100, 101, 102],
            "onch_diff_change_pct": [1.0, 1.0, 1.0],
        },
        index=idx,
    )
    return fee, bc, diff


def test_backfill_onchain_history(tmp_path, monkeypatch):
    start = "2021-01-01 00:02"
    end = "2021-01-01 00:12"
    idx = pd.date_range("2021-01-01 00:00", periods=3, freq="5min", tz="UTC")
    fee, bc, diff = _make_df(idx)
    bc = bc.iloc[0:0]

    monkeypatch.setattr(mod, "fetch_hoenicke_fees", lambda s, e, sess: fee)
    monkeypatch.setattr(mod, "fetch_blockchain_mempool", lambda s, e, sess: bc)
    monkeypatch.setattr(mod, "fetch_mining_difficulty", lambda s, e, sess: diff)
    monkeypatch.setattr(
        mod,
        "fetch_current_snapshot",
        lambda ts, sess: pd.DataFrame(index=pd.DatetimeIndex([], tz="UTC")),
    )

    db_path = tmp_path / "test.sqlite"
    mod.backfill_onchain_history(start, end, str(db_path))

    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query("SELECT * FROM onchain_5m ORDER BY ts_utc", conn)
    conn.close()

    assert list(df["ts_utc"]) == [1609459200, 1609459500, 1609459800, 1609460100]
    assert df["onch_fee_fast_satvb"].isna().iloc[:2].all()
    assert df.loc[0, "onch_fee_wavg_satvb"] == 1.0
