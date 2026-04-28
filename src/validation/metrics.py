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
    pair_id: str | None = None
    rl_team_id: str | None = None
    opponent_team_id: str | None = None

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
    total_steps = max(steps, 1)

    metrics = {
        "validation/episodes": float(total),
        "validation/wins": float(wins),
        "validation/losses": float(losses),
        "validation/win_rate": float(wins / total),
        "validation/avg_battle_length": float(steps / total),
        "validation/avg_total_reward": float(total_reward / total),
        "validation/fallback_events_per_battle": float(fallback_events / total),
        "validation/fallback_events_per_step": float(fallback_events / total_steps),
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

    metrics.update(_aggregate_group_metrics(results, "pair_id", "pair"))
    metrics.update(_aggregate_group_metrics(results, "rl_team_id", "team"))

    return metrics


def build_validation_diagnostics(results: List[BattleResult]) -> Dict[str, Any]:
    """Build report-only diagnostics that are too detailed for MLflow scalars."""
    by_pair = _group_summaries(results, "pair_id")
    by_team = _group_summaries(results, "rl_team_id")
    fallback_episodes = [
        result.to_dict()
        for result in sorted(
            results,
            key=lambda item: (item.fallback_events, item.steps),
            reverse=True,
        )
        if result.fallback_events > 0
    ]

    return {
        "by_pair": by_pair,
        "by_rl_team": by_team,
        "fallback": {
            "episodes_with_fallbacks": float(
                sum(1 for result in results if result.fallback_events > 0)
            ),
            "max_fallback_events": float(
                max((result.fallback_events for result in results), default=0)
            ),
            "top_episodes": fallback_episodes[:10],
        },
    }


def _aggregate_group_metrics(
    results: List[BattleResult],
    attr: str,
    label: str,
) -> Dict[str, float]:
    summaries = _group_summaries(results, attr)
    if not summaries:
        return {}

    win_rates = [float(summary["win_rate"]) for summary in summaries.values()]
    fallback_rates = [
        float(summary["fallback_events_per_battle"]) for summary in summaries.values()
    ]
    metrics = {
        f"validation/{label}_win_rate_mean": float(sum(win_rates) / len(win_rates)),
        f"validation/{label}_win_rate_min": float(min(win_rates)),
        f"validation/{label}_win_rate_max": float(max(win_rates)),
        f"validation/{label}_fallback_events_per_battle_mean": float(
            sum(fallback_rates) / len(fallback_rates)
        ),
    }

    for group_id, summary in summaries.items():
        metrics[f"validation/{label}/{group_id}/win_rate"] = float(summary["win_rate"])
        metrics[f"validation/{label}/{group_id}/episodes"] = float(summary["episodes"])

    return metrics


def _group_summaries(
    results: List[BattleResult],
    attr: str,
) -> Dict[str, Dict[str, float]]:
    groups: Dict[str, List[BattleResult]] = {}
    for result in results:
        group_id = getattr(result, attr)
        if group_id is None:
            continue
        groups.setdefault(str(group_id), []).append(result)

    summaries: Dict[str, Dict[str, float]] = {}
    for group_id, group_results in sorted(groups.items()):
        total = len(group_results)
        wins = sum(1 for result in group_results if result.outcome == 1)
        losses = sum(1 for result in group_results if result.outcome == 0)
        fallbacks = sum(result.fallback_events for result in group_results)
        steps = sum(result.steps for result in group_results)
        summaries[group_id] = {
            "episodes": float(total),
            "wins": float(wins),
            "losses": float(losses),
            "win_rate": float(wins / total) if total else 0.0,
            "avg_battle_length": float(steps / total) if total else 0.0,
            "fallback_events": float(fallbacks),
            "fallback_events_per_battle": float(fallbacks / total) if total else 0.0,
        }
    return summaries
