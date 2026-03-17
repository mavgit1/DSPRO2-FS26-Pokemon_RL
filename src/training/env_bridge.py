from typing import Any, Callable, Dict, List

from src.config.TM_optimal_config import CurriculumStageConfig


def foreach_env(algo: Any, fn: Callable[[Any], Any]) -> list[Any]:
    def _call_on_worker(worker: Any) -> list[Any]:
        if hasattr(worker, "foreach_env"):
            return worker.foreach_env(fn)
        if hasattr(worker, "env"):
            env_obj = worker.env
            return [fn(env_obj)] if env_obj is not None else []
        return []

    runner_attr = getattr(algo, "env_runner_group", None)
    try:
        runner_group = runner_attr() if callable(runner_attr) else runner_attr
    except (TypeError, ValueError):
        runner_group = None
    if runner_group is not None and hasattr(runner_group, "foreach_worker"):
        return runner_group.foreach_worker(_call_on_worker)

    workers_attr = getattr(algo, "workers", None)
    try:
        workers_group = workers_attr() if callable(workers_attr) else workers_attr
    except (TypeError, ValueError):
        workers_group = None
    if workers_group is not None and hasattr(workers_group, "foreach_worker"):
        return workers_group.foreach_worker(_call_on_worker)

    return []


def collect_recent_outcomes(algo: Any) -> List[int]:
    nested = foreach_env(
        algo,
        lambda e: e.pop_recent_outcomes() if hasattr(e, "pop_recent_outcomes") else [],
    )
    outcomes: List[int] = []
    if not nested:
        return outcomes

    for worker_item in nested:
        if not isinstance(worker_item, list):
            continue
        for env_item in worker_item:
            if isinstance(env_item, list):
                outcomes.extend(int(v) for v in env_item if v in {0, 1})
    return outcomes


def collect_recent_episode_stats(algo: Any) -> List[Dict[str, float]]:
    nested = foreach_env(
        algo,
        lambda e: e.pop_recent_episode_stats()
        if hasattr(e, "pop_recent_episode_stats")
        else [],
    )
    stats: List[Dict[str, float]] = []
    if not nested:
        return stats
    for worker_item in nested:
        if not isinstance(worker_item, list):
            continue
        for env_item in worker_item:
            if isinstance(env_item, list):
                for item in env_item:
                    if isinstance(item, dict):
                        stats.append(item)
    return stats


def collect_env_memory_sentinels(algo: Any) -> Dict[str, float]:
    nested = foreach_env(
        algo,
        lambda e: e.get_memory_counters() if hasattr(e, "get_memory_counters") else {},
    )
    if not nested:
        return {}

    values_by_key: Dict[str, List[float]] = {}
    for worker_item in nested:
        if not isinstance(worker_item, list):
            continue
        for env_item in worker_item:
            if not isinstance(env_item, dict):
                continue
            for key, value in env_item.items():
                if not isinstance(value, (int, float)):
                    continue
                values_by_key.setdefault(str(key), []).append(float(value))

    metrics: Dict[str, float] = {}
    for key, values in values_by_key.items():
        if not values:
            continue
        metric_key = key.replace("/", "_")
        metrics[f"leak/{metric_key}_sum"] = float(sum(values))
        metrics[f"leak/{metric_key}_mean"] = float(sum(values) / len(values))
        metrics[f"leak/{metric_key}_max"] = float(max(values))
    return metrics


def apply_curriculum_stage(algo: Any, stage: CurriculumStageConfig) -> None:
    payload = stage.to_dict()
    foreach_env(
        algo,
        lambda e: e.apply_curriculum_stage(payload)
        if hasattr(e, "apply_curriculum_stage")
        else None,
    )
