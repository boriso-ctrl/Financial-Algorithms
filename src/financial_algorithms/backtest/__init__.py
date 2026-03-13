"""Backtest engines, blending utilities, and metrics."""

from .engine import run_backtest
from .blender import blend_signals
from .metrics import compute_metrics
from .strategy_registry import StrategyRegistry, registry

__all__ = [
    "run_backtest",
    "blend_signals",
    "compute_metrics",
    "StrategyRegistry",
    "registry",
]
