"""Configuration loading helpers."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from crypto_analyzer.config.schema import (
    AppConfig,
    BacktestSettings,
    CalibrationSettings,
    CoreSettings,
    CVSettings,
    DatabaseSettings,
    DerivativeDataSettings,
    ExecutionSettings,
    FeatureSettings,
    ModelSettings,
    OnChainSettings,
    OrderbookSettings,
    RuntimeSettings,
    SentimentSettings,
)
from crypto_analyzer.utils.errors import ConfigError
from crypto_analyzer.utils.secrets import get_secret, load_environment

CONFIG_FILE_ENV = "APP_CONFIG_FILE"

_TRUE_VALUES = {"1", "true", "yes", "on"}
_FALSE_VALUES = {"0", "false", "no", "off"}

load_environment()


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
    text = str(value).strip()
    return text if text else default


def _as_int(value: Any, default: int) -> int:
    if value is None:
        return default
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return default


def _as_float(value: Any, default: float) -> float:
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
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


def _as_list(value: Any, default: list[Any]) -> list[Any]:
    if value is None:
        return list(default)
    if isinstance(value, (list, tuple, set)):
        return list(value)
    return [value]


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
    return default if limit == 0 else limit


def _build_core_settings(data: dict[str, Any]) -> CoreSettings:
    defaults = CoreSettings()
    symbol = os.getenv("SYMBOL") or _as_str(data.get("symbol"), defaults.symbol)
    interval = os.getenv("INTERVAL") or _as_str(data.get("interval"), defaults.interval)
    forward_env = os.getenv("FORWARD_STEPS")
    forward_steps = (
        _as_int(forward_env, defaults.forward_steps)
        if forward_env is not None
        else _as_int(data.get("forward_steps"), defaults.forward_steps)
    )
    history_days = _as_int(data.get("history_days"), defaults.history_days)
    timezone = _as_str(data.get("timezone"), defaults.timezone)
    return CoreSettings(
        symbol=symbol,
        interval=interval,
        forward_steps=forward_steps,
        history_days=history_days,
        timezone=timezone,
    )


def _build_database_settings(data: dict[str, Any]) -> DatabaseSettings:
    defaults = DatabaseSettings()
    db_path = os.getenv("DB_PATH") or _as_str(data.get("price_store"), str(defaults.price_store))
    db_url = (
        os.getenv("DATABASE_URL") or os.getenv("DB_URL") or _as_str(data.get("url"), defaults.url)
    )
    table_pred = os.getenv("TABLE_PRED") or _as_str(
        data.get("predictions_table"), defaults.predictions_table
    )
    onchain_table = _as_str(data.get("onchain_table"), defaults.onchain_table)
    feature_store_value = data.get("feature_store")
    feature_store = (
        str(feature_store_value)
        if feature_store_value not in (None, "")
        else defaults.feature_store
    )
    read_chunksize = _as_int(data.get("read_chunksize"), defaults.read_chunksize)
    pool_size = _as_int(os.getenv("DB_POOL_SIZE"), defaults.pool_size)
    max_overflow = _as_int(os.getenv("DB_MAX_OVERFLOW"), defaults.max_overflow)
    return DatabaseSettings(
        price_store=Path(db_path),
        url=db_url,
        predictions_table=table_pred,
        onchain_table=onchain_table,
        feature_store=Path(feature_store) if feature_store else None,
        read_chunksize=read_chunksize,
        pool_size=pool_size,
        max_overflow=max_overflow,
    )


def _build_runtime_settings(data: dict[str, Any]) -> RuntimeSettings:
    defaults = RuntimeSettings()
    cpu_env = os.getenv("CPU_LIMIT")
    cpu_limit = _resolve_cpu_limit(cpu_env if cpu_env is not None else data.get("cpu_limit"))
    repeat_env = os.getenv("REPEAT_COUNT")
    repeat_count = (
        _as_int(repeat_env, defaults.repeat_count)
        if repeat_env is not None
        else _as_int(data.get("repeat_count"), defaults.repeat_count)
    )
    log_level = _as_str(data.get("log_level"), defaults.log_level)
    data_dir = Path(_as_str(data.get("data_dir"), str(defaults.data_dir)))
    cache_dir = Path(_as_str(data.get("cache_dir"), str(defaults.cache_dir)))
    tmp_dir = Path(_as_str(data.get("tmp_dir"), str(defaults.tmp_dir)))
    orderbook_interval = _as_int(
        data.get("orderbook_interval_minutes"), defaults.orderbook_interval_minutes
    )
    binance_interval = _as_int(
        data.get("binance_interval_minutes"), defaults.binance_interval_minutes
    )
    return RuntimeSettings(
        cpu_limit=cpu_limit,
        repeat_count=repeat_count,
        log_level=log_level,
        data_dir=data_dir,
        cache_dir=cache_dir,
        tmp_dir=tmp_dir,
        orderbook_interval_minutes=orderbook_interval,
        binance_interval_minutes=binance_interval,
    )


def _build_feature_settings(
    data: dict[str, Any], sentiment: SentimentSettings | None = None
) -> FeatureSettings:
    defaults = FeatureSettings()
    default_sentiment = (
        sentiment.use_sentiment if sentiment is not None else defaults.include_sentiment
    )
    include_onchain = _as_bool(data.get("include_onchain"), defaults.include_onchain)
    include_orderbook = _as_bool(data.get("include_orderbook"), defaults.include_orderbook)
    include_derivatives = _as_bool(data.get("include_derivatives"), defaults.include_derivatives)
    include_sentiment = _as_bool(data.get("include_sentiment"), default_sentiment)
    forward_fill_limit = _as_int(data.get("forward_fill_limit"), defaults.forward_fill_limit)
    fillna_value = _as_float(data.get("fillna_value"), defaults.fillna_value)
    return FeatureSettings(
        include_onchain=include_onchain,
        include_orderbook=include_orderbook,
        include_derivatives=include_derivatives,
        include_sentiment=include_sentiment,
        forward_fill_limit=forward_fill_limit,
        fillna_value=fillna_value,
    )


def _build_live_feature_settings(
    data: dict[str, Any], sentiment: SentimentSettings | None = None
) -> FeatureSettings:
    return _build_feature_settings(data, sentiment=sentiment)


def _build_model_settings(data: dict[str, Any]) -> ModelSettings:
    defaults = ModelSettings()
    directory = Path(_as_str(data.get("directory"), str(defaults.directory)))
    weights_glob = _as_str(data.get("weights_glob"), defaults.weights_glob)
    use_gpu = _as_bool(data.get("use_gpu"), defaults.use_gpu)
    gpu_tree_method = _as_str(data.get("gpu_tree_method"), defaults.gpu_tree_method)
    max_models = _as_int(data.get("max_models"), defaults.max_models)
    random_seed = _as_int(data.get("random_seed"), defaults.random_seed)
    return ModelSettings(
        directory=directory,
        weights_glob=weights_glob,
        use_gpu=use_gpu,
        gpu_tree_method=gpu_tree_method,
        max_models=max_models,
        random_seed=random_seed,
    )


def _build_backtest_settings(data: dict[str, Any]) -> BacktestSettings:
    defaults = BacktestSettings()
    mode = _as_str(data.get("mode"), defaults.mode)
    validation_fraction = _as_float(data.get("validation_fraction"), defaults.validation_fraction)
    walkforward_window_days = _as_int(
        data.get("walkforward_window_days"), defaults.walkforward_window_days
    )
    metrics = tuple(str(item) for item in _as_list(data.get("metrics"), list(defaults.metrics)))
    return BacktestSettings(
        mode=mode,
        validation_fraction=validation_fraction,
        walkforward_window_days=walkforward_window_days,
        metrics=metrics,
    )


def _build_onchain_settings(data: dict[str, Any], runtime: RuntimeSettings) -> OnChainSettings:
    defaults = OnChainSettings()
    use_mempool = _as_bool(data.get("use_mempool"), defaults.use_mempool)
    use_exchange_flows = _as_bool(data.get("use_exchange_flows"), defaults.use_exchange_flows)
    use_usdt_events = _as_bool(data.get("use_usdt_events"), defaults.use_usdt_events)
    cache_default = runtime.cache_dir / "onchain"
    cache_dir = Path(_as_str(data.get("cache_dir"), str(cache_default)))
    glassnode_secret = get_secret("GLASSNODE_API_KEY")
    whale_secret = get_secret("WHALE_API_KEY")
    if glassnode_secret is not None:
        glassnode = glassnode_secret
    else:
        glassnode_api_key = data.get("glassnode_api_key")
        glassnode = (
            str(glassnode_api_key).strip() or None if glassnode_api_key is not None else None
        )

    if whale_secret is not None:
        whale = whale_secret
    else:
        whale_api_key = data.get("whale_api_key")
        whale = str(whale_api_key).strip() or None if whale_api_key is not None else None
    exchange_flow_source = _as_str(data.get("exchange_flow_source"), defaults.exchange_flow_source)
    exchange_flow_path_value = data.get("exchange_flow_path")
    if exchange_flow_path_value in (None, ""):
        exchange_flow_path = None
    else:
        exchange_flow_path = Path(str(exchange_flow_path_value))
    request_timeout = _as_int(data.get("request_timeout"), defaults.request_timeout)
    request_retries = _as_int(data.get("request_retries"), defaults.request_retries)
    return OnChainSettings(
        use_mempool=use_mempool,
        use_exchange_flows=use_exchange_flows,
        use_usdt_events=use_usdt_events,
        cache_dir=cache_dir,
        glassnode_api_key=glassnode,
        whale_api_key=whale,
        exchange_flow_source=exchange_flow_source,
        exchange_flow_path=exchange_flow_path,
        request_timeout=request_timeout,
        request_retries=request_retries,
    )


def _build_cv_settings(data: dict[str, Any]) -> CVSettings:
    defaults = CVSettings()
    cv_type = _as_str(data.get("type"), defaults.type)
    embargo = _as_int(data.get("embargo_min"), defaults.embargo_min)
    n_splits = _as_int(data.get("n_splits"), defaults.n_splits)
    return CVSettings(type=cv_type, embargo_min=embargo, n_splits=n_splits)


def _build_calibration_settings(data: dict[str, Any]) -> CalibrationSettings:
    defaults = CalibrationSettings()
    method = _as_str(data.get("method"), defaults.method).lower()
    return CalibrationSettings(method=method)


def _build_execution_settings(data: dict[str, Any]) -> ExecutionSettings:
    defaults = ExecutionSettings()
    fees_bps = _as_float(data.get("fees_bps"), defaults.fees_bps)
    slip_bps = _as_float(data.get("slip_bps"), defaults.slip_bps)
    latency = _as_float(data.get("latency_min"), defaults.latency_min)
    return ExecutionSettings(fees_bps=fees_bps, slip_bps=slip_bps, latency_min=latency)


def _build_derivative_settings(data: dict[str, Any]) -> DerivativeDataSettings:
    defaults = DerivativeDataSettings()
    funding_source = data.get("funding_source")
    basis_source = data.get("basis_source")
    oi_source = data.get("open_interest_source")
    freq = _as_str(data.get("resample_freq"), defaults.resample_freq)
    return DerivativeDataSettings(
        funding_source=Path(funding_source) if funding_source not in (None, "") else None,
        basis_source=Path(basis_source) if basis_source not in (None, "") else None,
        open_interest_source=Path(oi_source) if oi_source not in (None, "") else None,
        resample_freq=freq,
    )


def _build_orderbook_settings(data: dict[str, Any]) -> OrderbookSettings:
    defaults = OrderbookSettings()
    depth_levels = _as_int(data.get("depth_levels"), defaults.depth_levels)
    if depth_levels <= 0:
        depth_levels = defaults.depth_levels
    return OrderbookSettings(depth_levels=depth_levels)


def _build_sentiment_settings(data: dict[str, Any]) -> SentimentSettings:
    defaults = SentimentSettings()
    use_sentiment = _as_bool(data.get("use_sentiment"), defaults.use_sentiment)
    source = _as_str(data.get("sentiment_source"), defaults.sentiment_source)
    if source not in {"api", "csv"}:
        source = defaults.sentiment_source
    api_key_raw = data.get("sentiment_api_key")
    if api_key_raw in (None, ""):
        api_key = None
    else:
        api_key = str(api_key_raw)
    return SentimentSettings(
        use_sentiment=use_sentiment,
        sentiment_source=source,  # type: ignore[arg-type]
        sentiment_api_key=api_key,
    )


def _build_config() -> AppConfig:
    raw_config, path = _read_config_file()
    try:
        core = _build_core_settings(raw_config.get("core", {}))
        database = _build_database_settings(raw_config.get("database", {}))
        runtime = _build_runtime_settings(raw_config.get("runtime", {}))
        sentiment = _build_sentiment_settings(raw_config.get("sentiment", {}))
        features = _build_feature_settings(raw_config.get("features", {}), sentiment=sentiment)
        live_features = _build_live_feature_settings(raw_config.get("live", {}), sentiment=sentiment)
        models = _build_model_settings(raw_config.get("models", {}))
        backtest = _build_backtest_settings(raw_config.get("backtest", {}))
        onchain = _build_onchain_settings(raw_config.get("onchain", {}), runtime)
        cv = _build_cv_settings(raw_config.get("cv", {}))
        calibration = _build_calibration_settings(raw_config.get("calibration", {}))
        execution = _build_execution_settings(raw_config.get("execution", {}))
        derivatives = _build_derivative_settings(raw_config.get("derivatives", {}))
        orderbook = _build_orderbook_settings(raw_config.get("orderbook", {}))
        horizons = tuple(int(float(x)) for x in _as_list(raw_config.get("horizons"), []))
        live_universe = tuple(
            str(item).strip().upper()
            for item in _as_list(raw_config.get("live_universe"), [])
            if str(item).strip()
        )
        live_asset_limits = raw_config.get("live_asset_limits", {}) or {}
        live_initial_equity = _as_float(raw_config.get("live_initial_equity"), 10_000.0)
        live_initial_cash = _as_float(raw_config.get("live_initial_cash"), 10_000.0)
        live_position_size = _as_float(raw_config.get("live_position_size"), 0.01)
        live_target_volatility = _as_float(raw_config.get("live_target_volatility"), 0.02)
        live_min_position_size = _as_float(raw_config.get("live_min_position_size"), 0.001)
        live_max_position_size = _as_float(raw_config.get("live_max_position_size"), 0.05)
        if not horizons:
            default_horizon = core.forward_steps * 15
            horizons = (default_horizon,)
        pct_threshold = _as_float(raw_config.get("pct_threshold"), 0.01)
    except Exception as exc:  # pragma: no cover - defensive
        raise ConfigError(f"Unable to parse configuration: {exc}") from exc

    try:
        config = AppConfig(
            core=core,
            database=database,
            runtime=runtime,
            features=features,
            live=live_features,
            sentiment=sentiment,
            models=models,
            backtest=backtest,
            onchain=onchain,
            cv=cv,
            calibration=calibration,
            execution=execution,
            horizons=horizons,
            pct_threshold=pct_threshold,
            derivatives=derivatives,
            orderbook=orderbook,
            live_universe=live_universe,
            live_asset_limits=live_asset_limits,
            live_initial_equity=live_initial_equity,
            live_initial_cash=live_initial_cash,
            live_position_size=live_position_size,
            live_target_volatility=live_target_volatility,
            live_min_position_size=live_min_position_size,
            live_max_position_size=live_max_position_size,
            config_path=path,
        )
    except ValidationError as exc:  # pragma: no cover - defensive
        raise ConfigError(f"Invalid configuration: {exc}") from exc
    return config


CONFIG = _build_config()


def override_feature_settings(
    settings: FeatureSettings,
    *,
    include_onchain: bool | None = None,
    include_orderbook: bool | None = None,
    include_derivatives: bool | None = None,
    include_sentiment: bool | None = None,
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
    if include_sentiment is not None:
        updates["include_sentiment"] = bool(include_sentiment)
    if forward_fill_limit is not None:
        updates["forward_fill_limit"] = int(forward_fill_limit)
    if fillna_value is not None:
        updates["fillna_value"] = float(fillna_value)
    if not updates:
        return settings
    return settings.model_copy(update=updates)


def config_to_dict(config: AppConfig) -> dict[str, Any]:
    """Return a JSON/YAML serialisable representation of *config*."""

    return config.model_dump(mode="json")


__all__ = [
    "AppConfig",
    "BacktestSettings",
    "CONFIG",
    "CoreSettings",
    "DatabaseSettings",
    "FeatureSettings",
    "ModelSettings",
    "OnChainSettings",
    "SentimentSettings",
    "RuntimeSettings",
    "override_feature_settings",
    "config_to_dict",
]
