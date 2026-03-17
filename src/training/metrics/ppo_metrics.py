from typing import Any, Dict

from src.training.metrics.common import collect_numeric_values_for_exact_keys


def collect_ppo_metrics(result: Dict[str, Any]) -> Dict[str, float]:
    learner_payload = result.get("learners")
    if learner_payload is None:
        learner_payload = result.get("info", {}).get("learner")
    if learner_payload is None:
        learner_payload = result

    alias_map = {
        "ppo/policy_loss": ["policy_loss", "pi_loss", "mean_policy_loss"],
        "ppo/value_loss": ["vf_loss", "value_loss", "mean_vf_loss", "critic_loss"],
        "ppo/entropy": ["entropy", "entropy_loss", "mean_entropy"],
        "ppo/kl": ["kl", "mean_kl_loss", "kl_loss"],
        "ppo/explained_variance": ["explained_variance", "vf_explained_var"],
        "ppo/clip_fraction": ["clip_frac", "clipped", "clip_fraction"],
    }

    out: Dict[str, float] = {}
    for metric_name, aliases in alias_map.items():
        values = collect_numeric_values_for_exact_keys(learner_payload, aliases)
        if values:
            out[metric_name] = float(sum(values) / len(values))
    return out
