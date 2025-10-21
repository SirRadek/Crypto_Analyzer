"""APScheduler integration for recurring data collection jobs."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime, time, timedelta
from typing import Any, Callable

import pandas as pd

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.schedulers.base import BaseScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger

from crypto_analyzer.data.data_collector import (
    fetch_binance_funding_rates,
    fetch_binance_open_interest,
    fetch_binance_order_book,
    fetch_glassnode_active_addresses,
)
from crypto_analyzer.data.ingestion_store import (
    StoreResult,
    ensure_schema,
    latest_coinmetrics_timestamp,
    latest_derivatives_timestamp,
    latest_fear_greed_timestamp,
    latest_glassnode_timestamp,
    store_coinmetrics_flows,
    store_derivatives,
    store_fear_greed_index,
    store_glassnode_active_addresses,
    store_news,
    store_orderbook,
    store_reddit_sentiment,
)
from crypto_analyzer.data.news_fetcher import fetch_cryptopanic_news
from crypto_analyzer.data.onchain_fetcher import fetch_coinmetrics_exchange_flows
from crypto_analyzer.data.sentiment_index_fetcher import fetch_fear_greed_index
from crypto_analyzer.data.social_sentiment import fetch_reddit_sentiment
from crypto_analyzer.utils.logging import get_logger
from crypto_analyzer.utils.secrets import get_secret

try:  # pragma: no cover - optional during tests
    from crypto_analyzer.utils.config import CONFIG, AppConfig
except ModuleNotFoundError:  # pragma: no cover - fallback when config is optional
    CONFIG = None  # type: ignore[assignment]
    AppConfig = Any  # type: ignore[misc, assignment]

try:  # pragma: no cover - optional dependency on Python < 3.9
    from zoneinfo import ZoneInfo
except ImportError:  # pragma: no cover - fallback when zoneinfo missing
    from backports.zoneinfo import ZoneInfo  # type: ignore[no-redef]


LOGGER = get_logger(__name__)


@dataclass(slots=True)
class SchedulerConfig:
    """Settings controlling job cadence and API parameters."""

    binance_interval_minutes: int = 15
    news_interval_minutes: int = 15
    reddit_interval_minutes: int = 60
    daily_run_time: time = time(hour=0, minute=33)
    timezone: str | None = None
    retry_delay_seconds: int = 60
    max_retries: int = 1
    reddit_subreddit: str | None = "bitcoin"
    reddit_query: str | None = "BTC"
    reddit_size: int = 200
    cryptopanic_filter: str | None = None
    cryptopanic_currencies: tuple[str, ...] = field(default_factory=lambda: ("BTC",))
    cryptopanic_kind: str | None = None
    cryptopanic_limit: int = 50
    coinmetrics_asset: str = "btc"
    glassnode_asset: str = "BTC"
    binance_depth: int = 50
    daily_lookback_days: int = 30
    derivatives_initial_lookback_days: int = 3
    derivatives_overlap_hours: int = 6
    fear_greed_limit: int = 30


class DataIngestionScheduler:
    """Manage recurring data collection jobs using APScheduler."""

    def __init__(
        self,
        scheduler: BaseScheduler | None = None,
        *,
        settings: SchedulerConfig | None = None,
        app_config: AppConfig | None = None,
    ) -> None:
        self.settings = settings or SchedulerConfig()
        cfg = app_config
        if cfg is None:
            if CONFIG is None:  # pragma: no cover - defensive
                raise RuntimeError("Application configuration is unavailable")
            cfg = CONFIG
        self._config: AppConfig = cfg

        tz_name = self.settings.timezone or getattr(self._config.core, "timezone", "UTC")
        try:
            timezone = ZoneInfo(tz_name)
        except Exception:  # pragma: no cover - fallback to UTC when zone not found
            timezone = ZoneInfo("UTC")
        self.timezone = timezone

        if scheduler is None:
            scheduler = BackgroundScheduler(timezone=timezone)
        else:
            scheduler.configure(timezone=timezone)
        self.scheduler: BaseScheduler = scheduler

        self.logger = LOGGER
        self.engine = ensure_schema(db_path=str(self._config.db_path))

        sentiment_cfg = getattr(self._config, "sentiment", None)
        token = get_secret("CRYPTOPANIC_TOKEN")
        if not token and sentiment_cfg is not None:
            token = getattr(sentiment_cfg, "sentiment_api_key", None)
        self._cryptopanic_token: str | None = token if token else None

        onchain_cfg = getattr(self._config, "onchain", None)
        api_key = getattr(onchain_cfg, "glassnode_api_key", None) if onchain_cfg else None
        if not api_key:
            api_key = get_secret("GLASSNODE_API_KEY")
        self._glassnode_api_key = api_key

    # ------------------------------------------------------------------
    # Job scheduling helpers
    # ------------------------------------------------------------------

    def configure_jobs(self) -> None:
        """Register recurring jobs according to :class:`SchedulerConfig`."""

        if self.settings.binance_interval_minutes > 0:
            trigger = IntervalTrigger(minutes=self.settings.binance_interval_minutes, timezone=self.timezone)
            self.scheduler.add_job(
                self._wrap_job(self._run_binance_job, job_id="binance"),
                trigger,
                id="binance_job",
                replace_existing=True,
                max_instances=1,
                kwargs={"attempt": 1},
            )
        else:
            self.logger.info("Binance job disabled via configuration", extra={"job_id": "binance"})

        if self.settings.news_interval_minutes > 0:
            trigger = IntervalTrigger(minutes=self.settings.news_interval_minutes, timezone=self.timezone)
            self.scheduler.add_job(
                self._wrap_job(self._run_news_job, job_id="news"),
                trigger,
                id="news_job",
                replace_existing=True,
                max_instances=1,
                kwargs={"attempt": 1},
            )

        if self.settings.reddit_interval_minutes > 0 and (
            self.settings.reddit_subreddit or self.settings.reddit_query
        ):
            trigger = IntervalTrigger(minutes=self.settings.reddit_interval_minutes, timezone=self.timezone)
            self.scheduler.add_job(
                self._wrap_job(self._run_reddit_job, job_id="reddit"),
                trigger,
                id="reddit_job",
                replace_existing=True,
                max_instances=1,
                kwargs={"attempt": 1},
            )
        else:
            self.logger.info(
                "Reddit job disabled or lacking parameters",
                extra={"job_id": "reddit"},
            )

        daily_time = self.settings.daily_run_time
        cron = CronTrigger(hour=daily_time.hour, minute=daily_time.minute, timezone=self.timezone)
        self.scheduler.add_job(
            self._wrap_job(self._run_daily_job, job_id="daily"),
            cron,
            id="daily_job",
            replace_existing=True,
            max_instances=1,
            kwargs={"attempt": 1},
        )

    def start(self) -> None:
        """Start the underlying APScheduler instance."""

        self.logger.info("Starting data ingestion scheduler", extra={"timezone": str(self.timezone)})
        self.scheduler.start()

    def shutdown(self, *, wait: bool = True) -> None:
        """Stop the scheduler and cancel scheduled retries."""

        self.logger.info("Shutting down data ingestion scheduler")
        self.scheduler.shutdown(wait=wait)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _wrap_job(self, func: Callable[[], None], *, job_id: str) -> Callable[[int], None]:
        def runner(attempt: int = 1) -> None:
            extra = {"job_id": job_id, "attempt": attempt}
            self.logger.info("Starting job", extra=extra)
            try:
                func()
            except Exception as exc:  # pragma: no cover - network failures
                self.logger.error("Job execution failed", exc_info=exc, extra=extra)
                if attempt > self.settings.max_retries:
                    self.logger.error("Maximum retry attempts reached", extra=extra)
                    return
                run_time = datetime.now(tz=self.timezone) + timedelta(seconds=self.settings.retry_delay_seconds)
                retry_id = f"{job_id}_retry_{attempt}"
                retry_extra = {"job_id": job_id, "attempt": attempt + 1, "run_time": run_time.isoformat()}
                self.logger.info("Scheduling retry", extra=retry_extra)
                self.scheduler.add_job(
                    runner,
                    trigger="date",
                    run_date=run_time,
                    id=retry_id,
                    replace_existing=True,
                    kwargs={"attempt": attempt + 1},
                )
            else:
                self.logger.info("Job completed", extra=extra)

        return runner

    # ------------------------------------------------------------------
    # Job implementations
    # ------------------------------------------------------------------

    def _run_binance_job(self) -> None:
        symbol = self._config.symbol
        now = datetime.now(tz=UTC)
        latest = latest_derivatives_timestamp(self.engine, symbol=symbol)
        if latest is not None:
            start = latest.tz_convert("UTC") - timedelta(hours=max(1, self.settings.derivatives_overlap_hours))
        else:
            start = pd.Timestamp(now - timedelta(days=max(1, self.settings.derivatives_initial_lookback_days)), tz="UTC")
        if start > pd.Timestamp(now):
            start = pd.Timestamp(now - timedelta(minutes=5), tz="UTC")
        start_dt = start.to_pydatetime()

        self.logger.info(
            "Fetching Binance metrics",
            extra={"job_id": "binance", "symbol": symbol, "start": start_dt.isoformat(), "end": now.isoformat()},
        )

        funding = fetch_binance_funding_rates(symbol, start=start_dt, end=now)
        open_interest = fetch_binance_open_interest(symbol, start=start_dt, end=now)
        derivatives_result = store_derivatives(funding, open_interest, engine=self.engine, symbol=symbol)

        orderbook = fetch_binance_order_book(symbol, depth=self.settings.binance_depth)
        orderbook_result = store_orderbook(orderbook, engine=self.engine, symbol=symbol)

        self._log_store_result(
            "binance",
            {
                "derivative_rows": derivatives_result.inserted,
                "orderbook_rows": orderbook_result.inserted,
            },
        )

    def _run_news_job(self) -> None:
        if not self._cryptopanic_token:
            self.logger.warning("CryptoPanic token missing; skipping news job", extra={"job_id": "news"})
            return

        currencies = [code for code in self.settings.cryptopanic_currencies if code]
        limit = max(1, min(100, self.settings.cryptopanic_limit))
        kwargs: dict[str, Any] = {
            "limit": limit,
            "currencies": currencies if currencies else None,
            "filter": self.settings.cryptopanic_filter,
            "kind": self.settings.cryptopanic_kind,
        }
        self.logger.info("Fetching CryptoPanic news", extra={"job_id": "news", "limit": kwargs["limit"]})
        news_frame = fetch_cryptopanic_news(self._cryptopanic_token, **kwargs)
        result = store_news(news_frame, engine=self.engine)
        self._log_store_result("news", {"records": result.inserted})

    def _run_reddit_job(self) -> None:
        subreddit = self.settings.reddit_subreddit or ""
        query = self.settings.reddit_query or ""
        self.logger.info(
            "Fetching Reddit sentiment",
            extra={"job_id": "reddit", "subreddit": subreddit, "query": query},
        )
        size = max(20, min(500, self.settings.reddit_size))
        sentiment = fetch_reddit_sentiment(
            subreddit=subreddit or None,
            query=query or None,
            size=size,
        )
        result = store_reddit_sentiment(sentiment, engine=self.engine, subreddit=subreddit, query=query)
        self._log_store_result("reddit", {"records": result.inserted})

    def _run_daily_job(self) -> None:
        now = datetime.now(tz=UTC)
        asset_glassnode = self.settings.glassnode_asset
        asset_coinmetrics = self.settings.coinmetrics_asset

        if self._glassnode_api_key:
            latest_glassnode = latest_glassnode_timestamp(self.engine, asset=asset_glassnode)
            if latest_glassnode is not None:
                start_glassnode = latest_glassnode.tz_convert("UTC") - timedelta(days=7)
            else:
                start_glassnode = pd.Timestamp(now - timedelta(days=self.settings.daily_lookback_days), tz="UTC")
            start_glassnode = min(start_glassnode, pd.Timestamp(now))
            self.logger.info(
                "Fetching Glassnode active addresses",
                extra={"job_id": "daily", "asset": asset_glassnode, "start": start_glassnode.isoformat(), "end": now.isoformat()},
            )
            glassnode = fetch_glassnode_active_addresses(
                start=start_glassnode.to_pydatetime(),
                end=now,
                api_key=self._glassnode_api_key,
                asset=asset_glassnode,
            )
            glassnode_result = store_glassnode_active_addresses(
                glassnode, engine=self.engine, asset=asset_glassnode
            )
        else:
            self.logger.warning(
                "Glassnode API key missing; skipping active address download",
                extra={"job_id": "daily"},
            )
            glassnode_result = StoreResult(0)

        latest_coinmetrics = latest_coinmetrics_timestamp(self.engine, asset=asset_coinmetrics)
        if latest_coinmetrics is not None:
            start_coinmetrics = latest_coinmetrics.tz_convert("UTC") - timedelta(days=7)
        else:
            start_coinmetrics = pd.Timestamp(now - timedelta(days=self.settings.daily_lookback_days), tz="UTC")
        start_coinmetrics = min(start_coinmetrics, pd.Timestamp(now))
        self.logger.info(
            "Fetching CoinMetrics flows",
            extra={"job_id": "daily", "asset": asset_coinmetrics, "start": start_coinmetrics.isoformat(), "end": now.isoformat()},
        )
        coinmetrics = fetch_coinmetrics_exchange_flows(
            asset=asset_coinmetrics,
            start=start_coinmetrics,
            end=pd.Timestamp(now, tz="UTC"),
        )
        coinmetrics_result = store_coinmetrics_flows(
            coinmetrics, engine=self.engine, asset=asset_coinmetrics
        )

        self.logger.info(
            "Fetching Fear & Greed index",
            extra={"job_id": "daily", "limit": self.settings.fear_greed_limit},
        )
        fear_greed = fetch_fear_greed_index(limit=self.settings.fear_greed_limit)
        latest_seen = latest_fear_greed_timestamp(self.engine)
        if latest_seen is not None and not fear_greed.empty:
            fear_greed = fear_greed.loc[fear_greed["timestamp"] > latest_seen]
        fear_greed_result = store_fear_greed_index(fear_greed, engine=self.engine)

        self._log_store_result(
            "daily",
            {
                "glassnode_rows": glassnode_result.inserted,
                "coinmetrics_rows": coinmetrics_result.inserted,
                "fear_greed_rows": fear_greed_result.inserted,
            },
        )

    def _log_store_result(self, job_id: str, metrics: dict[str, Any]) -> None:
        payload = {"job_id": job_id}
        payload.update(metrics)
        self.logger.info("Persisted fetched data", extra=payload)


__all__ = ["DataIngestionScheduler", "SchedulerConfig"]
