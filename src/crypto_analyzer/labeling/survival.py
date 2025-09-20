"""Discrete time survival style labelling utilities.

The default classification labels focus on a fixed horizon (``touch ±0.5 %``
within N minutes).  When trading models are stacked in a survival setup we
often need the hazard of the price touching the barrier at *each* step.  This
module implements that logic in a vectorised fashion using pandas/NumPy and
is shared by both the model training code and reporting notebooks.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

__all__ = [
    "HazardConfig",
    "hazard_touch",
    "time_to_event",
]


@dataclass(slots=True)
class HazardConfig:
    """Describe the touch-hazard labelling problem."""

    horizon: int = 24
    barrier: float = 0.005
    direction: Literal["both", "up", "down"] = "both"

    def __post_init__(self) -> None:
        if self.horizon <= 0:
            raise ValueError("horizon must be positive")
        if self.barrier <= 0:
            raise ValueError("barrier must be positive")


def _direction_mask(direction: str, up: np.ndarray, down: np.ndarray) -> np.ndarray:
    if direction == "both":
        return up | down
    if direction == "up":
        return up
    if direction == "down":
        return down
    raise ValueError(f"Unsupported direction '{direction}'")


def hazard_touch(prices: pd.Series, *, config: HazardConfig | None = None) -> pd.DataFrame:
    """Return discrete hazards for touching the barrier within the horizon.

    Parameters
    ----------
    prices:
        Price series sampled at a fixed interval.
    config:
        Optional :class:`HazardConfig`.  The default corresponds to a 2 hour
        window on 5 minute candles (24 steps).
    """

    cfg = config or HazardConfig()
    if not isinstance(prices, pd.Series):
        prices = pd.Series(prices)
    prices = prices.astype(float)

    ratios = pd.DataFrame(index=prices.index)
    hazards: dict[str, np.ndarray] = {}
    alive = np.ones(len(prices), dtype=bool)

    for step in range(1, cfg.horizon + 1):
        future = prices.shift(-step) / prices - 1.0
        up = future >= cfg.barrier
        down = future <= -cfg.barrier
        event = _direction_mask(cfg.direction, up.to_numpy(dtype=bool), down.to_numpy(dtype=bool))
        hazard = event & alive
        hazards[f"hazard_step_{step}"] = hazard.astype(float)
        alive &= ~event

    for column, values in hazards.items():
        ratios[column] = values

    return ratios


def time_to_event(hazards: pd.DataFrame) -> pd.Series:
    """Convert hazard indicators into the first event index (hitting time)."""

    if hazards.empty:
        return pd.Series(index=hazards.index, dtype=float)

    steps = np.arange(1, len(hazards.columns) + 1, dtype=float)
    hazard_values = hazards.to_numpy(dtype=float)
    # cumulative hazard: once an event happens subsequent steps remain 0
    hit = hazard_values * steps
    hit[hazard_values == 0] = np.nan
    first = np.nanmin(hit, axis=1)
    return pd.Series(first, index=hazards.index, name="time_to_event")

