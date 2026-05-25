from dataclasses import asdict, dataclass
import math
import re
from typing import Any, Dict, List, Tuple


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
        metrics["validation/attack_action_ratio"] = float(
            attack_actions / total_actions
        )
        metrics["validation/switch_action_ratio"] = float(
            switch_actions / total_actions
        )

    by_opponent: Dict[str, List[BattleResult]] = {}
    missing_opponent_type_count = 0
    for result in results:
        opponent = _canonical_opponent_type(result.opponent_type)
        if opponent is None:
            missing_opponent_type_count += 1
            continue
        by_opponent.setdefault(opponent, []).append(result)

    if missing_opponent_type_count:
        metrics["validation/missing_opponent_type_count"] = float(
            missing_opponent_type_count
        )

    for opponent, opponent_results in by_opponent.items():
        opponent_total = len(opponent_results)
        opponent_wins = sum(1 for result in opponent_results if result.outcome == 1)
        metrics[f"validation/win_rate_vs_{opponent}"] = float(
            opponent_wins / opponent_total
        )
        metrics[f"validation/episodes_vs_{opponent}"] = float(opponent_total)

    return metrics


def _canonical_opponent_type(value: str | None) -> str | None:
    if value is None:
        return None
    key = str(value).strip().lower()
    if not key or key == "unknown":
        return None
    if key == "heuristics":
        return "heuristic"
    key = re.sub(r"[^a-z0-9_.-]+", "_", key)
    key = re.sub(r"_+", "_", key).strip("_")
    return key or None


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


def wilson_score_interval(
    wins: int, total: int, z: float = 1.96
) -> Tuple[float, float]:
    """95% Wilson score confidence interval for a binomial proportion.

    Returns (lower, upper) bounds in [0, 1].
    """
    if total == 0:
        return 0.0, 0.0
    p_hat = wins / total
    n = total
    denom = 1.0 + z * z / n
    centre = p_hat + z * z / (2.0 * n)
    spread = z * math.sqrt(p_hat * (1.0 - p_hat) / n + z * z / (4.0 * n * n))
    lower = max(0.0, (centre - spread) / denom)
    upper = min(1.0, (centre + spread) / denom)
    return lower, upper


_SKILL_WEIGHTS: Dict[str, float] = {
    "random": 1.0,
    "random_no_switch": 1.5,
    "heuristic": 2.0,
}


def compute_benchmark_metrics(results: List[BattleResult]) -> Dict[str, float]:
    """Compute benchmark metrics with per-opponent CIs and composite scores.

    Returns flat MLflow-safe scalars including:
    - ``benchmark/win_rate_vs_<opponent>`` per opponent
    - ``benchmark/ci_lower_vs_<opponent>`` / ``ci_upper_vs_<opponent>``
    - ``benchmark/skill_score`` — weighted average across opponent tiers
    - ``benchmark/consistency`` — fraction of tiers with WR > 0.5
    """
    metrics: Dict[str, float] = {}

    by_opponent: Dict[str, List[BattleResult]] = {}
    for result in results:
        key = _canonical_opponent_type(result.opponent_type)
        if key:
            by_opponent.setdefault(key, []).append(result)

    weighted_sum = 0.0
    weight_total = 0.0
    tiers_above_half = 0

    for opponent, opponent_results in by_opponent.items():
        n = len(opponent_results)
        wins = sum(1 for r in opponent_results if r.outcome == 1)
        wr = wins / n if n else 0.0
        lower, upper = wilson_score_interval(wins, n)

        metrics[f"benchmark/win_rate_vs_{opponent}"] = wr
        metrics[f"benchmark/ci_lower_vs_{opponent}"] = lower
        metrics[f"benchmark/ci_upper_vs_{opponent}"] = upper
        metrics[f"benchmark/episodes_vs_{opponent}"] = float(n)

        weight = _SKILL_WEIGHTS.get(opponent, 1.0)
        weighted_sum += wr * weight
        weight_total += weight
        if wr > 0.5:
            tiers_above_half += 1

    total_opponents = len(by_opponent)
    metrics["benchmark/skill_score"] = (
        weighted_sum / weight_total if weight_total > 0 else 0.0
    )
    metrics["benchmark/consistency"] = (
        tiers_above_half / total_opponents if total_opponents > 0 else 0.0
    )
    metrics["benchmark/total_episodes"] = float(len(results))
    metrics["benchmark/total_wins"] = float(sum(1 for r in results if r.outcome == 1))
    metrics["benchmark/overall_win_rate"] = (
        float(sum(1 for r in results if r.outcome == 1) / len(results))
        if results
        else 0.0
    )

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
