from dataclasses import dataclass
import os


@dataclass(frozen=True)
class Config:
    """Central application configuration loaded from environment."""

    symbol: str = os.getenv("SYMBOL", "BTCUSDT")
    db_path: str = os.getenv("DB_PATH", "db/data/crypto_data.sqlite")
    table_pred: str = os.getenv("TABLE_PRED", "prediction")
    interval: str = os.getenv("INTERVAL", "5m")
    forward_steps: int = int(os.getenv("FORWARD_STEPS", "24"))


CONFIG = Config()

__all__ = ["CONFIG"]
