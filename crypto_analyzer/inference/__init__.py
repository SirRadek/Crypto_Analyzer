"""Inference helpers with lazy GPU support."""
from __future__ import annotations

from ._gpu import get_gpu_rf_or_none

__all__ = ["get_gpu_rf_or_none"]
