"""Tests for TimescaleDB schema helpers."""

from __future__ import annotations

import logging
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock

import pytest


class _SQLText(str):
    def format(self, *args, **kwargs):
        return _SQLText(self)


class _SQLModule:
    def SQL(self, text: str) -> _SQLText:
        return _SQLText(text)

    def Identifier(self, value: str) -> str:
        return value


_psycopg2_stub = SimpleNamespace(
    OperationalError=RuntimeError,
    sql=_SQLModule(),
    connect=MagicMock(),
)

sys.modules.setdefault("psycopg2", _psycopg2_stub)
sys.modules.setdefault("psycopg2.sql", _psycopg2_stub.sql)
sys.modules.setdefault(
    "psycopg2.extensions", SimpleNamespace(connection=object)
)

_config_module = ModuleType("crypto_analyzer.utils.config")
_config_module.CONFIG = SimpleNamespace(
    interval="1h",
    database=SimpleNamespace(url=None),
)
sys.modules.setdefault("crypto_analyzer.utils.config", _config_module)

from crypto_analyzer.data import db


def _mock_cursor(conn: MagicMock) -> MagicMock:
    cursor = conn.cursor.return_value.__enter__.return_value
    return cursor


def test_dependency_check_raises_for_missing_tables() -> None:
    conn = MagicMock()
    cursor = _mock_cursor(conn)
    cursor.fetchall.return_value = [("market_data",), ("news",)]

    with pytest.raises(RuntimeError) as excinfo:
        db._ensure_combined_features_dependencies(conn)

    assert "missing tables" in str(excinfo.value)


def test_create_materialized_view_runs_all_statements(monkeypatch) -> None:
    conn = MagicMock()
    cursor = _mock_cursor(conn)
    cursor.fetchall.return_value = [(name,) for name in db.COMBINED_FEATURES_DEPENDENCIES]

    executed: list[str] = []

    def capture_statements(_: MagicMock, statements: tuple[str, ...]) -> None:
        executed.extend(statements)

    monkeypatch.setattr(db, "_execute_statements", capture_statements)

    db._create_combined_features_materialized_view(conn)

    assert executed[0] == "DROP MATERIALIZED VIEW IF EXISTS combined_features"
    assert executed[1] == "DROP VIEW IF EXISTS combined_features"
    assert db.COMBINED_FEATURES_MATERIALIZED_VIEW in executed
    assert db.COMBINED_FEATURES_INDEX in executed


def test_refresh_combined_features_concurrently() -> None:
    conn = MagicMock()
    cursor = _mock_cursor(conn)

    db.refresh_combined_features(conn, concurrently=True)

    cursor.execute.assert_called_once_with(
        "REFRESH MATERIALIZED VIEW CONCURRENTLY combined_features"
    )
    conn.commit.assert_called_once()


def test_refresh_combined_features_logs_and_rolls_back(caplog: pytest.LogCaptureFixture) -> None:
    conn = MagicMock()
    cursor = _mock_cursor(conn)
    cursor.execute.side_effect = RuntimeError("lock contention")

    caplog.set_level(logging.ERROR)

    with pytest.raises(RuntimeError):
        db.refresh_combined_features(conn)

    conn.rollback.assert_called_once()
    assert "Failed to refresh combined_features materialized view" in caplog.text
