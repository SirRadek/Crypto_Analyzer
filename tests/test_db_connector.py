from __future__ import annotations

from pathlib import Path

import pandas as pd

from crypto_analyzer.data import db_connector


def _build_rows() -> list[list[float]]:
    return [
        [
            1_700_000_000_000,
            100.0,
            110.0,
            90.0,
            105.0,
            1_000.0,
            1_700_000_050_000,
            2_000.0,
            1_234,
            500.0,
            600.0,
        ],
        [
            1_700_000_600_000,
            105.0,
            115.0,
            95.0,
            110.0,
            1_200.0,
            1_700_000_650_000,
            2_100.0,
            1_300,
            550.0,
            650.0,
        ],
    ]


def test_save_and_get_price_data(tmp_path: Path) -> None:
    db_url = f"sqlite:///{tmp_path / 'prices.sqlite'}"
    symbol = "BTCUSDT"
    interval = "1h"

    db_connector.init_timescale(db_path=db_url)
    db_connector.save_to_db(_build_rows(), symbol, interval, db_path=db_url, batch_size=1)

    updated_first = _build_rows()[0].copy()
    updated_first[4] = 106.0
    updated_first[8] = 1_500
    db_connector.save_to_db([updated_first], symbol, interval, db_path=db_url)

    frame = db_connector.get_price_data(symbol, db_path=db_url)
    assert len(frame) == 2
    assert pd.api.types.is_datetime64tz_dtype(frame["timestamp"])
    assert frame.loc[0, "close"] == 106.0
    assert frame.loc[0, "number_of_trades"] == 1_500
    assert (
        db_connector.get_latest_open_time(symbol=symbol, interval=interval, db_path=db_url)
        == 1_700_000_600_000
    )
