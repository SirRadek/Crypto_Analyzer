from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field


class FeatureConfig(BaseModel):
    path: Path


class TrainConfig(BaseModel):
    horizon_min: int = Field(..., gt=0)
    embargo: int = Field(..., ge=0)
    target_kind: Literal["log", "lin"] = "log"
    xgb_params: dict[str, dict[str, Any]]
    bounds: dict[str, float] | None = None
    fees: dict[str, float] | None = None
    features: FeatureConfig
    n_jobs: int = 4


class PredictionRequest(BaseModel):
    features: dict[str, float]


class PredictionResponse(BaseModel):
    timestamp: str
    p_hat: float

    model_config = {"validate_assignment": True}
