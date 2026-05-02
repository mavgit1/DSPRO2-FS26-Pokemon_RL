"""
Validation utilities for gauntlet construction and evaluation.
"""

from src.validation.metrics import (
    BattleResult,
    aggregate_validation_metrics,
    build_validation_diagnostics,
)
from src.validation.protocols import ValidationProtocol, get_protocol
from src.validation.runner import run_validation

__all__ = [
    "BattleResult",
    "ValidationProtocol",
    "aggregate_validation_metrics",
    "build_validation_diagnostics",
    "get_protocol",
    "run_validation",
]

