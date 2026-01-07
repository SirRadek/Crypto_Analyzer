"""Core trading types for paper execution."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Literal

Side = Literal["buy", "sell"]
OrderType = Literal["limit", "market"]


@dataclass(slots=True)
class Order:
    symbol: str
    side: Side
    quantity: float
    price: float
    order_type: OrderType = "limit"
    ts: datetime | None = None


@dataclass(slots=True)
class Position:
    symbol: str
    quantity: float
    entry_price: float
    leverage: float


@dataclass(slots=True)
class Portfolio:
    equity: float
    cash: float
    positions: dict[str, Position]
