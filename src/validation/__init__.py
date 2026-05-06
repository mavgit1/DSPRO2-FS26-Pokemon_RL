"""
Validation utilities for battle evaluation and benchmarking.
"""

from src.validation.metrics import (
    BattleResult,
    aggregate_validation_metrics,
    build_validation_diagnostics,
    compute_benchmark_metrics,
    wilson_score_interval,
)
from src.validation.protocols import ValidationProtocol, get_protocol
from src.validation.reporting import format_validation_summary
from src.validation.runner import run_inprocess_validation, run_validation

__all__ = [
    "BattleResult",
    "ValidationProtocol",
    "aggregate_validation_metrics",
    "build_validation_diagnostics",
    "compute_benchmark_metrics",
    "format_validation_summary",
    "get_protocol",
    "run_inprocess_validation",
    "run_validation",
    "wilson_score_interval",
]
