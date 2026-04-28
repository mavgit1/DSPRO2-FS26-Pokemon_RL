from dataclasses import asdict, dataclass
from typing import Any, Dict, List


@dataclass
class BattleResult:
    """Result for one validation battle."""

    episode: int
    opponent_type: str
    outcome: int
    total_reward: float
    steps: int
    fallback_events: int = 0
    attack_actions: int = 0
    switch_actions: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def aggregate_validation_metrics(results: List[BattleResult]) -> Dict[str, float]:
    """Aggregate battle-level validation results into MLflow-safe scalars."""
    if not results:
        return {
            "validation/episodes": 0.0,
            "validation/win_rate": 0.0,
        }

    total = len(results)
    wins = sum(1 for result in results if result.outcome == 1)
    losses = sum(1 for result in results if result.outcome == 0)
    steps = sum(result.steps for result in results)
    fallback_events = sum(result.fallback_events for result in results)
    attack_actions = sum(result.attack_actions for result in results)
    switch_actions = sum(result.switch_actions for result in results)
    total_actions = attack_actions + switch_actions
    total_reward = sum(result.total_reward for result in results)

    metrics = {
        "validation/episodes": float(total),
        "validation/wins": float(wins),
        "validation/losses": float(losses),
        "validation/win_rate": float(wins / total),
        "validation/avg_battle_length": float(steps / total),
        "validation/avg_total_reward": float(total_reward / total),
        "validation/fallback_events_per_battle": float(fallback_events / total),
    }
    if total_actions > 0:
        metrics["validation/attack_action_ratio"] = float(attack_actions / total_actions)
        metrics["validation/switch_action_ratio"] = float(switch_actions / total_actions)

    by_opponent: Dict[str, List[BattleResult]] = {}
    for result in results:
        by_opponent.setdefault(result.opponent_type, []).append(result)

    for opponent, opponent_results in by_opponent.items():
        opponent_total = len(opponent_results)
        opponent_wins = sum(1 for result in opponent_results if result.outcome == 1)
        metrics[f"validation/win_rate_vs_{opponent}"] = float(
            opponent_wins / opponent_total
        )
        metrics[f"validation/episodes_vs_{opponent}"] = float(opponent_total)

    return metrics
