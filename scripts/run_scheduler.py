"""Entry point for running the automated data ingestion scheduler."""

from __future__ import annotations

import argparse
import sys
from contextlib import suppress

from apscheduler.schedulers.blocking import BlockingScheduler

from crypto_analyzer.scheduling import DataIngestionScheduler, SchedulerConfig
from crypto_analyzer.utils.logging import configure_logging, resolve_run_id


def _parse_args() -> SchedulerConfig:
    parser = argparse.ArgumentParser(description="Start the Crypto Analyzer data scheduler")
    parser.add_argument(
        "--timezone",
        help="Timezone override for scheduled jobs (defaults to config core.timezone)",
    )
    parser.add_argument(
        "--binance-interval",
        type=int,
        help="Minutes between Binance market data refreshes",
    )
    parser.add_argument(
        "--news-interval",
        type=int,
        help="Minutes between CryptoPanic news refreshes",
    )
    parser.add_argument(
        "--reddit-interval",
        type=int,
        help="Minutes between Reddit sentiment refreshes",
    )
    parser.add_argument(
        "--daily-time",
        help="Cron-style HH:MM time for daily aggregations",
    )
    parser.add_argument(
        "--retry-delay",
        type=int,
        help="Seconds to wait before retrying a failed job",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        help="Maximum retry attempts for a failed job",
    )
    args = parser.parse_args()

    settings = SchedulerConfig()
    if args.timezone:
        settings.timezone = args.timezone
    if args.binance_interval:
        settings.binance_interval_minutes = max(1, args.binance_interval)
    if args.news_interval:
        settings.news_interval_minutes = max(1, args.news_interval)
    if args.reddit_interval:
        settings.reddit_interval_minutes = max(1, args.reddit_interval)
    if args.retry_delay:
        settings.retry_delay_seconds = max(1, args.retry_delay)
    if args.max_retries is not None:
        settings.max_retries = max(0, args.max_retries)
    if args.daily_time:
        try:
            hour_str, minute_str = args.daily_time.split(":", 1)
            settings.daily_run_time = settings.daily_run_time.replace(
                hour=int(hour_str), minute=int(minute_str)
            )
        except (ValueError, AttributeError):
            raise SystemExit("--daily-time must be in HH:MM format")
    return settings


def main() -> None:
    configure_logging(resolve_run_id())
    settings = _parse_args()
    scheduler = BlockingScheduler()
    ingestion = DataIngestionScheduler(scheduler=scheduler, settings=settings)
    ingestion.configure_jobs()
    try:
        ingestion.start()
    except (KeyboardInterrupt, SystemExit):
        with suppress(Exception):
            ingestion.shutdown()
        sys.exit(0)
    except Exception:  # pragma: no cover - unexpected runtime failure
        with suppress(Exception):
            ingestion.shutdown(wait=False)
        raise


if __name__ == "__main__":  # pragma: no cover - manual execution
    main()
