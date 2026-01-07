"""Custom exception hierarchy used across CLI utilities."""

from __future__ import annotations


class CryptoAnalyzerError(Exception):
    """Base class for domain specific exceptions."""


class DataValidationError(CryptoAnalyzerError):
    """Raised when user supplied data or CLI arguments are invalid."""


class ConfigError(CryptoAnalyzerError):
    """Raised when configuration loading or validation fails."""


class ModelError(CryptoAnalyzerError):
    """Raised for model and training related failures."""


__all__ = ["ConfigError", "CryptoAnalyzerError", "DataValidationError", "ModelError"]
