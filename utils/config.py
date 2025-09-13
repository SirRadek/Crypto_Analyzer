import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    """Central application configuration loaded from environment."""

    symbol: str = os.getenv("SYMBOL", "BTCUSDT")
    db_path: str = os.getenv("DB_PATH", "db/data/crypto_data.sqlite")
    table_pred: str = os.getenv("TABLE_PRED", "predictions")
    interval: str = os.getenv("INTERVAL", "5m")
    forward_steps: int = int(os.getenv("FORWARD_STEPS", "24"))
    cpu_limit: int = int(os.getenv("CPU_LIMIT", str(os.cpu_count() or 20)))
    repeat_count: int = int(os.getenv("REPEAT_COUNT", "5"))


CONFIG = Config()

__all__ = ["CONFIG"]
