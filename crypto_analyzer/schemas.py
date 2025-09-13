from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator


class FeatureConfig(BaseModel):
    path: Path


class TrainConfig(BaseModel):
    horizon_min: int = Field(..., gt=0)
    embargo: int = Field(..., ge=0)
    target_kind: Literal["log", "lin"] = "log"
    xgb_params: dict[str, dict[str, Any]]
    bounds: dict[str, float]
    fees: dict[str, float] | None = None
    features: FeatureConfig
    n_jobs: int = 4


class PredictionRequest(BaseModel):
    features: dict[str, float]


class PredictionResponse(BaseModel):
    timestamp: str
    p_low: float
    p_hat: float
    p_high: float

    @model_validator(mode="after")
    def check_order(self) -> PredictionResponse:
        if not (self.p_low <= self.p_hat <= self.p_high):
            raise ValueError("p_low <= p_hat <= p_high must hold")
        return self

    model_config = {"validate_assignment": True}
