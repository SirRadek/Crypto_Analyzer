from __future__ import annotations

import os
from dataclasses import asdict, dataclass, is_dataclass, replace
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

CONFIG_FILE_ENV = "APP_CONFIG_FILE"

_TRUE_VALUES = {"1", "true", "yes", "on"}
_FALSE_VALUES = {"0", "false", "no", "off"}

load_dotenv(override=False)


@dataclass(frozen=True)
class CoreSettings:
    symbol: str
    interval: str
    forward_steps: int
    history_days: int
    timezone: str


@dataclass(frozen=True)
class DatabaseSettings:
    price_store: str
    predictions_table: str
    onchain_table: str
    feature_store: str | None
    read_chunksize: int


@dataclass(frozen=True)
class RuntimeSettings:
    cpu_limit: int
    repeat_count: int
    log_level: str
    data_dir: str
    cache_dir: str
    tmp_dir: str


@dataclass(frozen=True)
class FeatureSettings:
    include_onchain: bool
    include_orderbook: bool
    include_derivatives: bool
    forward_fill_limit: int
    fillna_value: float


@dataclass(frozen=True)
class ModelSettings:
    directory: str
    weights_glob: str
    use_gpu: bool
    gpu_tree_method: str
    max_models: int
    random_seed: int


@dataclass(frozen=True)
class BacktestSettings:
    mode: str
    validation_fraction: float
    walkforward_window_days: int
    metrics: tuple[str, ...]


@dataclass(frozen=True)
class OnChainSettings:
    use_mempool: bool
    use_exchange_flows: bool
    use_usdt_events: bool
    cache_dir: str
    glassnode_api_key: str | None
    whale_api_key: str | None
    exchange_flow_source: str
    exchange_flow_path: str | None
    request_timeout: int
    request_retries: int


@dataclass(frozen=True)
class AppConfig:
    core: CoreSettings
    database: DatabaseSettings
    runtime: RuntimeSettings
    features: FeatureSettings
    models: ModelSettings
    backtest: BacktestSettings
    onchain: OnChainSettings
    config_path: Path | None = None

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
    def db_path(self) -> str:
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


def _read_config_file() -> tuple[dict[str, Any], Path | None]:
    """Load YAML configuration from file if available."""

    candidate = os.getenv(CONFIG_FILE_ENV)
    search_paths: list[Path] = []
    if candidate:
        search_paths.append(Path(candidate).expanduser())
    search_paths.append(Path("config/app.yaml"))
    search_paths.append(Path("config/app.example.yaml"))

    for path in search_paths:
        if not path.is_file():
            continue
        with path.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
        return data, path
    return {}, None


def _as_str(value: Any, default: str) -> str:
    if value is None:
        return default
    return str(value)


def _as_int(value: Any, default: int) -> int:
    if value is None:
        return default
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return default
        lowered = value.lower()
        if lowered in {"auto", "max", "all"}:
            return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _as_float(value: Any, default: float) -> float:
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return default
    return default


def _as_bool(value: Any, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in _TRUE_VALUES:
            return True
        if lowered in _FALSE_VALUES:
            return False
    try:
        return bool(int(value))
    except (TypeError, ValueError):
        return default


def _as_list(value: Any, default: list[str]) -> list[str]:
    if value is None:
        return list(default)
    if isinstance(value, (list, tuple, set)):
        return [str(item) for item in value]
    return [str(value)]


def _resolve_cpu_limit(value: Any) -> int:
    default = os.cpu_count() or -1
    if value is None:
        return default
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"", "auto", "max", "all"}:
            return default
    try:
        limit = int(value)
    except (TypeError, ValueError):
        return default
    return limit if limit != 0 else default


def _build_core_settings(data: dict[str, Any]) -> CoreSettings:
    symbol = os.getenv("SYMBOL") or _as_str(data.get("symbol"), "BTCUSDT")
    interval = os.getenv("INTERVAL") or _as_str(data.get("interval"), "15m")
    forward_env = os.getenv("FORWARD_STEPS")
    forward_steps = (
        int(forward_env)
        if forward_env is not None
        else _as_int(data.get("forward_steps"), 8)
    )
    history_days = _as_int(data.get("history_days"), 5 * 365)
    timezone = _as_str(data.get("timezone"), "UTC")
    return CoreSettings(
        symbol=symbol,
        interval=interval,
        forward_steps=forward_steps,
        history_days=history_days,
        timezone=timezone,
    )


def _build_database_settings(data: dict[str, Any]) -> DatabaseSettings:
    db_path = os.getenv("DB_PATH") or _as_str(
        data.get("price_store"), "data/crypto_data.sqlite"
    )
    table_pred = os.getenv("TABLE_PRED") or _as_str(
        data.get("predictions_table"), "predictions"
    )
    onchain_table = _as_str(data.get("onchain_table"), "onchain_15m")
    feature_store = data.get("feature_store")
    feature_store_str = str(feature_store) if feature_store not in (None, "") else None
    read_chunksize = _as_int(data.get("read_chunksize"), 100_000)
    return DatabaseSettings(
        price_store=db_path,
        predictions_table=table_pred,
        onchain_table=onchain_table,
        feature_store=feature_store_str,
        read_chunksize=read_chunksize,
    )


def _build_runtime_settings(data: dict[str, Any]) -> RuntimeSettings:
    cpu_env = os.getenv("CPU_LIMIT")
    cpu_limit = _resolve_cpu_limit(cpu_env if cpu_env is not None else data.get("cpu_limit"))
    repeat_env = os.getenv("REPEAT_COUNT")
    repeat_count = (
        int(repeat_env)
        if repeat_env is not None
        else _as_int(data.get("repeat_count"), 50)
    )
    log_level = _as_str(data.get("log_level"), "INFO")
    data_dir = _as_str(data.get("data_dir"), "data")
    cache_dir = _as_str(data.get("cache_dir"), "data/cache")
    tmp_dir = _as_str(data.get("tmp_dir"), "data/tmp")
    return RuntimeSettings(
        cpu_limit=cpu_limit,
        repeat_count=repeat_count,
        log_level=log_level,
        data_dir=data_dir,
        cache_dir=cache_dir,
        tmp_dir=tmp_dir,
    )


def _build_feature_settings(data: dict[str, Any]) -> FeatureSettings:
    include_onchain = _as_bool(data.get("include_onchain"), True)
    include_orderbook = _as_bool(data.get("include_orderbook"), True)
    include_derivatives = _as_bool(data.get("include_derivatives"), True)
    forward_fill_limit = _as_int(data.get("forward_fill_limit"), 12)
    fillna_value = _as_float(data.get("fillna_value"), 0.0)
    return FeatureSettings(
        include_onchain=include_onchain,
        include_orderbook=include_orderbook,
        include_derivatives=include_derivatives,
        forward_fill_limit=forward_fill_limit,
        fillna_value=fillna_value,
    )


def _build_model_settings(data: dict[str, Any]) -> ModelSettings:
    directory = _as_str(data.get("directory"), "artifacts/models")
    weights_glob = _as_str(data.get("weights_glob"), "artifacts/backtest_acc_*.json")
    use_gpu = _as_bool(data.get("use_gpu"), True)
    gpu_tree_method = _as_str(data.get("gpu_tree_method"), "gpu_hist")
    max_models = _as_int(data.get("max_models"), 16)
    random_seed = _as_int(data.get("random_seed"), 1337)
    return ModelSettings(
        directory=directory,
        weights_glob=weights_glob,
        use_gpu=use_gpu,
        gpu_tree_method=gpu_tree_method,
        max_models=max_models,
        random_seed=random_seed,
    )


def _build_backtest_settings(data: dict[str, Any]) -> BacktestSettings:
    mode = _as_str(data.get("mode"), "holdout")
    validation_fraction = _as_float(data.get("validation_fraction"), 0.2)
    walkforward_window_days = _as_int(data.get("walkforward_window_days"), 30)
    metrics = tuple(_as_list(data.get("metrics"), ["accuracy", "precision", "recall"]))
    return BacktestSettings(
        mode=mode,
        validation_fraction=validation_fraction,
        walkforward_window_days=walkforward_window_days,
        metrics=metrics,
    )


def _build_onchain_settings(
    data: dict[str, Any], runtime: RuntimeSettings
) -> OnChainSettings:
    use_mempool = _as_bool(data.get("use_mempool"), True)
    use_exchange_flows = _as_bool(data.get("use_exchange_flows"), True)
    use_usdt_events = _as_bool(data.get("use_usdt_events"), True)
    cache_default = Path(runtime.cache_dir) / "onchain"
    cache_dir = _as_str(data.get("cache_dir"), str(cache_default))
    glassnode_api_key = data.get("glassnode_api_key")
    if glassnode_api_key is not None:
        glassnode_api_key = str(glassnode_api_key) or None
    whale_api_key = data.get("whale_api_key")
    if whale_api_key is not None:
        whale_api_key = str(whale_api_key) or None
    exchange_flow_source = _as_str(data.get("exchange_flow_source"), "csv")
    exchange_flow_path_value = data.get("exchange_flow_path")
    exchange_flow_path = (
        str(exchange_flow_path_value)
        if exchange_flow_path_value not in (None, "")
        else None
    )
    request_timeout = _as_int(data.get("request_timeout"), 10)
    request_retries = _as_int(data.get("request_retries"), 5)
    return OnChainSettings(
        use_mempool=use_mempool,
        use_exchange_flows=use_exchange_flows,
        use_usdt_events=use_usdt_events,
        cache_dir=cache_dir,
        glassnode_api_key=glassnode_api_key,
        whale_api_key=whale_api_key,
        exchange_flow_source=exchange_flow_source,
        exchange_flow_path=exchange_flow_path,
        request_timeout=request_timeout,
        request_retries=request_retries,
    )


def _build_config() -> AppConfig:
    raw_config, path = _read_config_file()
    core = _build_core_settings(raw_config.get("core", {}))
    database = _build_database_settings(raw_config.get("database", {}))
    runtime = _build_runtime_settings(raw_config.get("runtime", {}))
    features = _build_feature_settings(raw_config.get("features", {}))
    models = _build_model_settings(raw_config.get("models", {}))
    backtest = _build_backtest_settings(raw_config.get("backtest", {}))
    onchain = _build_onchain_settings(raw_config.get("onchain", {}), runtime)
    return AppConfig(
        core=core,
        database=database,
        runtime=runtime,
        features=features,
        models=models,
        backtest=backtest,
        onchain=onchain,
        config_path=path,
    )


CONFIG = _build_config()

 
def override_feature_settings(
    settings: FeatureSettings,
    *,
    include_onchain: bool | None = None,
    include_orderbook: bool | None = None,
    include_derivatives: bool | None = None,
    forward_fill_limit: int | None = None,
    fillna_value: float | None = None,
) -> FeatureSettings:
    """Return updated feature settings with selected fields overridden."""

    updates: dict[str, object] = {}
    if include_onchain is not None:
        updates["include_onchain"] = bool(include_onchain)
    if include_orderbook is not None:
        updates["include_orderbook"] = bool(include_orderbook)
    if include_derivatives is not None:
        updates["include_derivatives"] = bool(include_derivatives)
    if forward_fill_limit is not None:
        updates["forward_fill_limit"] = int(forward_fill_limit)
    if fillna_value is not None:
        updates["fillna_value"] = float(fillna_value)
    if not updates:
        return settings
    return replace(settings, **updates)


__all__ = [
    "AppConfig",
    "BacktestSettings",
    "CONFIG",
    "CoreSettings",
    "DatabaseSettings",
    "FeatureSettings",
    "ModelSettings",
    "OnChainSettings",
    "RuntimeSettings",
    "override_feature_settings",
    "config_to_dict",
]


def _serialise(value: Any) -> Any:
    if is_dataclass(value):
        return {k: _serialise(v) for k, v in asdict(value).items()}
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return [_serialise(v) for v in value]
    if isinstance(value, list):
        return [_serialise(v) for v in value]
    if isinstance(value, dict):
        return {k: _serialise(v) for k, v in value.items()}
    return value


def config_to_dict(config: AppConfig) -> dict[str, Any]:
    """Return a JSON/YAML serialisable representation of *config*."""

    return _serialise(config)
