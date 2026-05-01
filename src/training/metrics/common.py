import json
import math
from typing import Any, Dict, List, Optional


def flatten_for_mlflow(prefix: str, value: Any, out: Dict[str, Any]) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            flatten_for_mlflow(child_prefix, item, out)
        return

    if isinstance(value, list):
        for idx, item in enumerate(value):
            child_prefix = f"{prefix}.{idx}" if prefix else str(idx)
            flatten_for_mlflow(child_prefix, item, out)
        return

    if isinstance(value, (str, int, float, bool)):
        out[prefix] = value
    else:
        out[prefix] = json.dumps(value, default=str)


def find_numeric_by_substring(container: Any, key_substring: str) -> Optional[float]:
    target = key_substring.lower()

    def _walk(obj: Any, prefix: str = "") -> Optional[float]:
        if isinstance(obj, dict):
            for key, val in obj.items():
                child_prefix = f"{prefix}.{key}" if prefix else str(key)
                found = _walk(val, child_prefix)
                if found is not None:
                    return found
            return None
        if isinstance(obj, list):
            for idx, val in enumerate(obj):
                child_prefix = f"{prefix}.{idx}" if prefix else str(idx)
                found = _walk(val, child_prefix)
                if found is not None:
                    return found
            return None
        if isinstance(obj, (int, float)) and target in prefix.lower():
            return float(obj)
        return None

    return _walk(container)


def collect_numeric_values_for_exact_keys(
    container: Any, keys: List[str]
) -> List[float]:
    keys_set = {k.lower() for k in keys}
    out: List[float] = []

    def _walk(obj: Any) -> None:
        if isinstance(obj, dict):
            for key, val in obj.items():
                if key.lower() in keys_set and isinstance(val, (int, float)):
                    out.append(float(val))
                _walk(val)
        elif isinstance(obj, list):
            for item in obj:
                _walk(item)

    _walk(container)
    return out


def mean(values: List[float]) -> Optional[float]:
    if not values:
        return None
    return float(sum(values) / len(values))


def sanitize_mlflow_metrics(metrics: Dict[str, Any]) -> Dict[str, float]:
    """Keep only finite float scalars — some tracking backends reject NaN/Inf or odd types."""
    out: Dict[str, float] = {}
    for key, value in metrics.items():
        if not isinstance(key, str):
            continue
        try:
            x = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(x):
            out[key] = x
    return out
