"""Pydantic models describing the application configuration schema."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class _BaseModel(BaseModel):
    """Helper base class providing common configuration."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class CoreSettings(_BaseModel):
    symbol: str = "BTCUSDT"
    interval: str = "1d"
    forward_steps: int = Field(default=8, ge=1)
    history_days: int = Field(default=5 * 365, ge=1)
    timezone: str = "UTC"


class DatabaseSettings(_BaseModel):
    price_store: Path = Path("data/crypto_data.sqlite")
    predictions_table: str = "predictions"
    onchain_table: str = "onchain_1d"
    feature_store: Path | None = None
    read_chunksize: int = Field(default=100_000, ge=1)


class RuntimeSettings(_BaseModel):
    cpu_limit: int = Field(default=-1)
    repeat_count: int = Field(default=50, ge=1)
    log_level: str = "INFO"
    data_dir: Path = Path("data")
    cache_dir: Path = Path("data/cache")
    tmp_dir: Path = Path("data/tmp")


class FeatureSettings(_BaseModel):
    include_onchain: bool = True
    include_orderbook: bool = True
    include_derivatives: bool = True
    include_sentiment: bool = False
    forward_fill_limit: int = Field(default=12)
    fillna_value: float = 0.0


class ModelSettings(_BaseModel):
    directory: Path = Path("artifacts/models")
    weights_glob: str = "artifacts/backtest_acc_*.json"
    use_gpu: bool = True
    gpu_tree_method: str = "gpu_hist"
    max_models: int = Field(default=16, ge=1)
    random_seed: int = Field(default=1337)


class BacktestSettings(_BaseModel):
    mode: str = "holdout"
    validation_fraction: float = Field(default=0.2, ge=0.0, le=1.0)
    walkforward_window_days: int = Field(default=30, ge=1)
    metrics: tuple[str, ...] = ("accuracy", "precision", "recall")


class OnChainSettings(_BaseModel):
    use_mempool: bool = True
    use_exchange_flows: bool = True
    use_usdt_events: bool = True
    cache_dir: Path = Path("data/cache/onchain")
    glassnode_api_key: str | None = None
    whale_api_key: str | None = None
    exchange_flow_source: Literal["csv", "api"] = "csv"
    exchange_flow_path: Path | None = None
    request_timeout: int = Field(default=10, ge=0)
    request_retries: int = Field(default=5, ge=0)


class SentimentSettings(_BaseModel):
    use_sentiment: bool = False
    sentiment_source: Literal["api", "csv"] = "api"
    sentiment_api_key: str | None = None


class CVSettings(_BaseModel):
    type: str = "holdout"
    embargo_min: int = Field(default=0, ge=0)
    n_splits: int = Field(default=5, ge=1)


class CalibrationSettings(_BaseModel):
    method: str = "none"


class ExecutionSettings(_BaseModel):
    fees_bps: float = Field(default=0.0)
    slip_bps: float = Field(default=0.0)
    latency_min: float = Field(default=0.0)

    @field_validator("fees_bps", mode="after")
    @classmethod
    def _ensure_non_negative(cls, value: float) -> float:
        if value < 0:
            raise ValueError("fees_bps must be non-negative")
        return value


class DerivativeDataSettings(_BaseModel):
    funding_source: Path | None = None
    basis_source: Path | None = None
    open_interest_source: Path | None = None
    resample_freq: str = "5T"


class OrderbookSettings(_BaseModel):
    depth_levels: int = Field(default=5, ge=1)


class AppConfig(_BaseModel):
    core: CoreSettings
    database: DatabaseSettings
    runtime: RuntimeSettings
    features: FeatureSettings
    sentiment: SentimentSettings
    models: ModelSettings
    backtest: BacktestSettings
    onchain: OnChainSettings
    cv: CVSettings
    calibration: CalibrationSettings
    execution: ExecutionSettings
    horizons: tuple[int, ...]
    pct_threshold: float
    derivatives: DerivativeDataSettings
    orderbook: OrderbookSettings
    config_path: Path | None = None

    @field_validator("horizons", mode="before")
    @classmethod
    def _coerce_horizons(cls, value: object) -> tuple[int, ...]:
        if value is None:
            return tuple()
        if isinstance(value, (list, tuple, set)):
            return tuple(int(v) for v in value)
        return (int(value),)

    @model_validator(mode="after")
    def _validate_ranges(self) -> "AppConfig":
        if not (0 < self.pct_threshold < 0.1):
            raise ValueError("pct_threshold must be between 0 and 0.1")
        if self.horizons:
            max_horizon = max(self.horizons)
        else:
            max_horizon = 0
        if self.cv.embargo_min < max_horizon:
            raise ValueError("cv.embargo_min must be greater than or equal to the maximum horizon")
        return self

    @property
    def symbol(self) -> str:
        return self.core.symbol

    @property
    def interval(self) -> str:
        return self.core.interval

    @property
    def forward_steps(self) -> int:
        return self.core.forward_steps

    @property
    def db_path(self) -> Path:
        return self.database.price_store

    @property
    def table_pred(self) -> str:
        return self.database.predictions_table

    @property
    def cpu_limit(self) -> int:
        return self.runtime.cpu_limit

    @property
    def repeat_count(self) -> int:
        return self.runtime.repeat_count


__all__ = [
    "AppConfig",
    "BacktestSettings",
    "CalibrationSettings",
    "CoreSettings",
    "CVSettings",
    "DatabaseSettings",
    "DerivativeDataSettings",
    "ExecutionSettings",
    "FeatureSettings",
    "ModelSettings",
    "OnChainSettings",
    "SentimentSettings",
    "OrderbookSettings",
    "RuntimeSettings",
]

