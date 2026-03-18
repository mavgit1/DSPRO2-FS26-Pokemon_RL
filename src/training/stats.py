from typing import Dict, List

import numpy as np


def compute_training_stats(
    rewards: List[float],
    lengths: List[int],
    window: int = 100,
) -> Dict[str, float]:
    """Compute windowed training statistics."""
    if not rewards:
        return {}

    recent_rewards = rewards[-window:]
    recent_lengths = lengths[-window:]

    return {
        "reward_mean": np.mean(recent_rewards),
        "reward_std": np.std(recent_rewards),
        "reward_min": np.min(recent_rewards),
        "reward_max": np.max(recent_rewards),
        "length_mean": np.mean(recent_lengths),
        "length_std": np.std(recent_lengths),
    }
