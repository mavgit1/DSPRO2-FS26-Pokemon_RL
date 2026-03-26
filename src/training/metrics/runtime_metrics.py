from typing import Any, Dict

from src.training.metrics.common import find_numeric_by_substring


def collect_runtime_metrics(
    result: Dict[str, Any],
    train_time_ms: float,
    steps_delta: int,
    wall_delta_s: float,
) -> Dict[str, float]:
    metrics: Dict[str, float] = {
        "sys/train_time_ms": float(train_time_ms),
        "sys/env_steps_delta": float(max(0, steps_delta)),
    }
    if wall_delta_s > 0:
        metrics["sys/env_steps_per_sec"] = float(max(0, steps_delta) / wall_delta_s)

    sample_time = find_numeric_by_substring(result, "sample_time_ms")
    learner_time = find_numeric_by_substring(result, "learner_update_time_ms")
    if sample_time is None:
        sample_time = find_numeric_by_substring(result, "sample_ms")
    if learner_time is None:
        learner_time = find_numeric_by_substring(result, "learn_time_ms")

    if sample_time is not None:
        metrics["sys/sample_time_ms"] = float(sample_time)
    if learner_time is not None:
        metrics["sys/learner_update_time_ms"] = float(learner_time)
    return metrics
