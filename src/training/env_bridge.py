from typing import Any, Callable, Dict, List, Optional, Tuple

from src.config.TM_optimal_config import CurriculumStageConfig

_UNWRAP_STEPS = 32


def _next_wrapped_env(cur: Any) -> Optional[Any]:
    nxt = getattr(cur, "env", None)
    if nxt is None or nxt is cur:
        return None
    return nxt


def _leaf_envs(env: Any) -> List[Any]:
    """Resolve concrete sub-envs from RLlib's vector stack (e.g. DictInfoToList → SyncVectorEnv)."""
    if env is None:
        return []
    seen: set[int] = set()
    cur: Any = env
    for _ in range(_UNWRAP_STEPS):
        if id(cur) in seen:
            break
        seen.add(id(cur))
        envs_attr = getattr(cur, "envs", None)
        if isinstance(envs_attr, (list, tuple)) and len(envs_attr) > 0:
            return list(envs_attr)
        nxt = _next_wrapped_env(cur)
        if nxt is None:
            break
        cur = nxt
    return [cur]


def _vector_call(
    env: Any,
    method: str,
    *args: Any,
    **kwargs: Any,
) -> Optional[List[Any]]:
    """Dispatch ``method`` on each sub-env via gymnasium ``VectorEnv.call``."""
    cur: Any = env
    seen: set[int] = set()
    for _ in range(_UNWRAP_STEPS):
        if id(cur) in seen:
            break
        seen.add(id(cur))
        call = getattr(cur, "call", None)
        if callable(call):
            try:
                return list(call(method, *args, **kwargs))
            except Exception:
                pass
        nxt = _next_wrapped_env(cur)
        if nxt is None:
            break
        cur = nxt
    return None


def _fallback_invoke(env: Any, method: str, args: Tuple[Any, ...], default: Any) -> Any:
    fn = getattr(env, method, None)
    return fn(*args) if callable(fn) else default


def foreach_env(
    algo: Any,
    fn: Callable[[Any], Any],
    *,
    vector_method: Optional[str] = None,
    vector_args: Tuple[Any, ...] = (),
    vector_kwargs: Optional[Dict[str, Any]] = None,
) -> list[Any]:
    vk = vector_kwargs or {}

    def _call_on_worker(worker: Any) -> list[Any]:
        if hasattr(worker, "foreach_env"):
            return worker.foreach_env(fn)
        env_obj = getattr(worker, "env", None)
        if env_obj is None:
            return []
        if vector_method:
            got = _vector_call(env_obj, vector_method, *vector_args, **vk)
            if got is not None:
                return got
        return [fn(e) for e in _leaf_envs(env_obj)]

    runner_attr = getattr(algo, "env_runner_group", None)
    try:
        runner_group = runner_attr() if callable(runner_attr) else runner_attr
    except (TypeError, ValueError):
        runner_group = None
    if runner_group is not None and hasattr(runner_group, "foreach_env_runner"):
        return runner_group.foreach_env_runner(_call_on_worker, local_env_runner=True)

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


def _nested_per_env_batches(nested: list[Any]):
    for worker_item in nested:
        if isinstance(worker_item, (list, tuple)):
            yield from worker_item


def _foreach_vector_env(
    algo: Any,
    method: str,
    args: Tuple[Any, ...] = (),
    *,
    fallback: Any,
) -> list[Any]:
    return foreach_env(
        algo,
        lambda e: _fallback_invoke(e, method, args, fallback),
        vector_method=method,
        vector_args=args,
    )


def collect_recent_outcomes(algo: Any) -> List[int]:
    nested = _foreach_vector_env(algo, "pop_recent_outcomes", fallback=[])
    outcomes: List[int] = []
    if not nested:
        return outcomes
    for env_item in _nested_per_env_batches(nested):
        if not isinstance(env_item, (list, tuple)):
            continue
        for v in env_item:
            try:
                iv = int(v)
            except (TypeError, ValueError):
                continue
            if iv in (0, 1):
                outcomes.append(iv)
    return outcomes


def collect_recent_episode_stats(algo: Any) -> List[Dict[str, float]]:
    nested = _foreach_vector_env(algo, "pop_recent_episode_stats", fallback=[])
    stats: List[Dict[str, float]] = []
    if not nested:
        return stats
    for env_item in _nested_per_env_batches(nested):
        if not isinstance(env_item, (list, tuple)):
            continue
        for item in env_item:
            if isinstance(item, dict):
                stats.append(item)
    return stats


def collect_recent_observation_samples(algo: Any, max_samples_per_env: int = 3) -> List[Dict[str, Any]]:
    nested = _foreach_vector_env(
        algo,
        "pop_recent_observation_samples",
        (int(max_samples_per_env),),
        fallback=[],
    )
    samples: List[Dict[str, Any]] = []
    if not nested:
        return samples
    for env_item in _nested_per_env_batches(nested):
        if not isinstance(env_item, (list, tuple)):
            continue
        for item in env_item:
            if isinstance(item, dict):
                samples.append(item)
    return samples


def collect_env_memory_sentinels(algo: Any) -> Dict[str, float]:
    nested = _foreach_vector_env(algo, "get_memory_counters", fallback={})
    if not nested:
        return {}

    values_by_key: Dict[str, List[float]] = {}
    for env_item in _nested_per_env_batches(nested):
        if not isinstance(env_item, dict):
            continue
        for key, value in env_item.items():
            if isinstance(value, (int, float)):
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
    _foreach_vector_env(
        algo,
        "apply_curriculum_stage",
        (payload,),
        fallback=None,
    )
