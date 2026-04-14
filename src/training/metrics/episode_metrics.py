from collections import defaultdict
from typing import Dict, List, Optional

from src.training.metrics.common import mean


def aggregate_episode_metrics(
    outcomes: List[int],
    episode_stats: List[Dict[str, float]],
    stage_name: Optional[str] = None,
) -> Dict[str, float]:
    metrics: Dict[str, float] = {}

    if outcomes:
        wins = sum(1 for x in outcomes if x == 1)
        losses = sum(1 for x in outcomes if x == 0)
        total = len(outcomes)
        metrics["outcome/wins_interval"] = float(wins)
        metrics["outcome/losses_interval"] = float(losses)
        metrics["outcome/draws_or_unknown_interval"] = float(max(0, total - wins - losses))
        metrics["outcome/win_rate_interval"] = float(wins / total) if total > 0 else 0.0

    if not episode_stats:
        return metrics

    # Per-opponent win rates
    by_opponent: Dict[str, List[int]] = defaultdict(list)
    for stat in episode_stats:
        opp_type = stat.get("opponent_type", "unknown")
        outcome = stat.get("outcome")
        if outcome is not None:
            by_opponent[opp_type].append(int(outcome))

    for opp_type, opp_outcomes in by_opponent.items():
        opp_wins = sum(opp_outcomes)
        opp_total = len(opp_outcomes)
        if opp_total > 0:
            metrics[f"outcome/win_rate_vs_{opp_type}"] = float(opp_wins / opp_total)
            metrics[f"outcome/wins_vs_{opp_type}"] = float(opp_wins)
            metrics[f"outcome/total_vs_{opp_type}"] = float(opp_total)

    reward_victory = [float(s["reward_victory_component"]) for s in episode_stats if "reward_victory_component" in s]
    reward_hp = [float(s["reward_hp_diff_component"]) for s in episode_stats if "reward_hp_diff_component" in s]
    reward_faint = [float(s["reward_faint_component"]) for s in episode_stats if "reward_faint_component" in s]
    reward_step = [float(s["reward_step_penalty_component"]) for s in episode_stats if "reward_step_penalty_component" in s]
    our_hp = [float(s["terminal_our_hp_remaining"]) for s in episode_stats if "terminal_our_hp_remaining" in s]
    opp_hp = [float(s["terminal_opp_hp_remaining"]) for s in episode_stats if "terminal_opp_hp_remaining" in s]
    faint_diff = [float(s["terminal_faint_diff"]) for s in episode_stats if "terminal_faint_diff" in s]
    turns = [float(s["battle_turns"]) for s in episode_stats if "battle_turns" in s]
    win_turns = [
        float(s["battle_turns"])
        for s in episode_stats
        if "battle_turns" in s and float(s.get("outcome", -1.0)) == 1.0
    ]
    mask_valid = [float(s["action_mask_valid_count_mean"]) for s in episode_stats if "action_mask_valid_count_mean" in s]
    total_actions = [float(s["episode_total_actions"]) for s in episode_stats if "episode_total_actions" in s]
    attack_actions = [float(s["episode_attack_actions"]) for s in episode_stats if "episode_attack_actions" in s]
    switch_actions = [float(s["episode_switch_actions"]) for s in episode_stats if "episode_switch_actions" in s]
    fallback_events = [float(s["episode_fallback_events"]) for s in episode_stats if "episode_fallback_events" in s]

    mean_map = {
        "reward/reward_victory_component_mean": reward_victory,
        "reward/reward_hp_diff_component_mean": reward_hp,
        "reward/reward_faint_component_mean": reward_faint,
        "reward/reward_step_penalty_component_mean": reward_step,
        "battle/terminal_our_hp_remaining_mean": our_hp,
        "battle/terminal_opp_hp_remaining_mean": opp_hp,
        "battle/terminal_faint_diff_mean": faint_diff,
        "battle/episode_turns_mean": turns,
        "battle/avg_turns_to_win": win_turns,
        "action/action_mask_valid_count_mean": mask_valid,
    }
    for metric_name, vals in mean_map.items():
        mean_val = mean(vals)
        if mean_val is not None:
            metrics[metric_name] = mean_val

    # Stage-scoped reward tagging
    if stage_name:
        total_reward = [
            float(s["reward_victory_component"]) + float(s.get("reward_hp_diff_component", 0.0))
            + float(s.get("reward_faint_component", 0.0)) + float(s.get("reward_step_penalty_component", 0.0))
            for s in episode_stats
            if "reward_victory_component" in s
        ]
        stage_reward_mean = mean(total_reward)
        if stage_reward_mean is not None:
            metrics[f"episode_reward_stage/{stage_name}"] = stage_reward_mean

    total_action_sum = float(sum(total_actions))
    if total_action_sum > 0.0:
        metrics["action/attack_action_ratio"] = float(sum(attack_actions) / total_action_sum)
        metrics["action/switch_action_ratio"] = float(sum(switch_actions) / total_action_sum)
        metrics["action/illegal_action_fallback_rate"] = float(
            sum(fallback_events) / total_action_sum
        )

    return metrics
