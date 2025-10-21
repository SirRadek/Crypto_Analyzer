from __future__ import annotations

from datetime import datetime
from unittest.mock import MagicMock

import pandas as pd
import pytest

from crypto_analyzer.data import timescale_writer as writer


@pytest.fixture
def mock_connection() -> MagicMock:
    conn = MagicMock()
    cursor = MagicMock()
    cursor.__enter__.return_value = cursor
    cursor.__exit__.return_value = False
    conn.cursor.return_value = cursor
    conn.autocommit = False
    return conn


def test_save_market_data_upserts_valid_rows(mock_connection: MagicMock, monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    def fake_execute_batch(cursor, sql, params, page_size=None):  # type: ignore[no-untyped-def]
        captured["sql"] = sql
        captured["params"] = params
        captured["page_size"] = page_size

    monkeypatch.setattr(writer, "execute_batch", fake_execute_batch)

    frame = pd.DataFrame(
        [
            {
                "timestamp": "2024-01-01T00:00:00Z",
                "open": 42000,
                "high": 43000,
                "low": 41000,
                "close": 42500,
                "volume": 123.4,
                "quote_volume": 98765.4,
                "trades": 123,
            }
        ]
    )

    rows = writer.save_market_data(
        frame,
        symbol="BTCUSDT",
        interval="1h",
        connection=mock_connection,
    )

    assert rows == 1
    assert captured["page_size"] == 200
    assert "INSERT INTO market_data" in str(captured["sql"])

    params = captured["params"]
    assert isinstance(params, list)
    row = params[0]
    assert row[1] == "BTCUSDT"
    assert row[2] == "1h"
    assert isinstance(row[0], datetime)
    assert row[3] == pytest.approx(42000.0)

    mock_connection.cursor.assert_called_once()
    mock_connection.commit.assert_called_once()
    mock_connection.rollback.assert_not_called()


def test_save_market_data_requires_symbol(mock_connection: MagicMock, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(writer, "execute_batch", MagicMock())

    frame = pd.DataFrame(
        [
            {
                "timestamp": "2024-01-01T00:00:00Z",
                "open": 1,
                "high": 2,
                "low": 0.5,
                "close": 1.5,
            }
        ]
    )

    with pytest.raises(ValueError):
        writer.save_market_data(frame, interval="5m", connection=mock_connection)

    writer.execute_batch.assert_not_called()  # type: ignore[attr-defined]


def test_save_sentiment_index_validates_range(mock_connection: MagicMock, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(writer, "execute_batch", MagicMock())

    payload = [{"timestamp": "2024-01-01T00:00:00Z", "value": 150}]

    with pytest.raises(ValueError):
        writer.save_sentiment_index(payload, connection=mock_connection)

    writer.execute_batch.assert_not_called()  # type: ignore[attr-defined]


def test_save_news_strips_optional_fields(mock_connection: MagicMock, monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    def fake_execute_batch(cursor, sql, params, page_size=None):  # type: ignore[no-untyped-def]
        captured["sql"] = sql
        captured["params"] = params

    monkeypatch.setattr(writer, "execute_batch", fake_execute_batch)

    payload = [
        {
            "timestamp": "2024-03-01T12:30:00Z",
            "title": " Example headline ",
            "url": "   ",
            "source": " Coindesk ",
            "sentiment": 0.25,
        }
    ]

    rows = writer.save_news(payload, connection=mock_connection)

    assert rows == 1
    params = captured["params"]
    row = params[0]
    assert row[1] == "Example headline"
    assert row[2] is None  # URL stripped to None
    assert row[3] == "Coindesk"

