"""
Observation pipeline validator.

Validates observation samples collected from live training environments.
Checks that the information flow from battle state → embedding → action mask
is correct and consistent.

Designed to be called each training iteration on observation samples collected
via ``collect_recent_observation_samples``.
"""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

from src.models.embedding import (
    WEATHER_LIST,
    NUM_TOKENS,
    TOKEN_DIM,
)
from src.action_space import (
    COMPRESSED_ACTION_SPACE_N,
    COMPRESSED_SWITCH_ACTIONS,
)


# Index constants for token layout
_GLOBAL_TOKEN = 0
_OUR_ACTIVE_TOKEN = 1
_FIRST_BENCH_TOKEN = 2
_OPP_ACTIVE_TOKEN = 7
_FIRST_OPP_BENCH_TOKEN = 8

# Feature offsets within a pokemon token
_PRESENCE_OFFSET = 0     # is_present
_IS_ACTIVE_OFFSET = 1    # is_active
_IS_FAINTED_OFFSET = 2   # is_fainted


def validate_observations(
    samples: List[Dict[str, Any]],
) -> Dict[str, float]:
    """Run all validation checks on a batch of observation samples.

    Returns a flat dict of metrics suitable for MLflow logging.
    """
    if not samples:
        return {"obs_val/samples_checked": 0.0}

    metrics: Dict[str, float] = {
        "obs_val/samples_checked": float(len(samples)),
    }

    checks = [
        _check_shapes,
        _check_no_nan,
        _check_active_tokens,
        _check_weather_populated,
        _check_action_mask_binary,
        _check_switch_mask_consistency,
        _check_bench_fainted_flags,
        _check_species_nonzero,
    ]

    for check_fn in checks:
        check_metrics = check_fn(samples)
        metrics.update(check_metrics)

    # Aggregate overall health
    fail_count = sum(
        1 for k, v in metrics.items()
        if k.startswith("obs_val/") and k.endswith("_fail") and v > 0
    )
    metrics["obs_val/checks_with_failures"] = float(fail_count)
    metrics["obs_val/total_checks"] = float(len(checks))

    return metrics


def _check_shapes(samples: List[Dict[str, Any]]) -> Dict[str, float]:
    """Verify all arrays have expected shapes."""
    key = "obs_val/shape_mismatch"
    mismatches = 0
    for s in samples:
        obs = s.get("obs")
        mask = s.get("action_mask")
        if obs is None or mask is None:
            mismatches += 1
            continue
        obs_arr = np.asarray(obs)
        mask_arr = np.asarray(mask)
        if obs_arr.shape != (NUM_TOKENS, TOKEN_DIM):
            mismatches += 1
        elif mask_arr.shape != (COMPRESSED_ACTION_SPACE_N,):
            mismatches += 1
    return {
        f"{key}_count": float(mismatches),
        f"{key}_fail": float(mismatches > 0),
    }


def _check_no_nan(samples: List[Dict[str, Any]]) -> Dict[str, float]:
    """Check for NaN values in observation arrays."""
    nan_count = 0
    total_nan = 0
    for s in samples:
        obs = np.asarray(s.get("obs", np.array([])))
        if obs.size > 0 and np.any(np.isnan(obs)):
            nan_count += 1
            total_nan += int(np.isnan(obs).sum())
    return {
        "obs_val/nan_samples": float(nan_count),
        "obs_val/nan_total_values": float(total_nan),
        "obs_val/nan_fail": float(nan_count > 0),
    }


def _check_active_tokens(samples: List[Dict[str, Any]]) -> Dict[str, float]:
    """Verify token 1 has is_active=1 and token 2+ have is_active=0."""
    violations = 0
    for s in samples:
        obs = np.asarray(s.get("obs"))
        if obs.size == 0:
            continue
        # Token 1 (our active) should have is_active=1
        if obs[_OUR_ACTIVE_TOKEN, _IS_ACTIVE_OFFSET] < 0.5:
            violations += 1
            continue
        # Token 2-6 should have is_active=0
        for t in range(_FIRST_BENCH_TOKEN, min(_FIRST_BENCH_TOKEN + 5, NUM_TOKENS)):
            if obs[t, _PRESENCE_OFFSET] > 0.5 and obs[t, _IS_ACTIVE_OFFSET] > 0.5:
                violations += 1
                break
    return {
        "obs_val/active_token_violations": float(violations),
        "obs_val/active_token_fail": float(violations > 0),
    }


def _check_weather_populated(samples: List[Dict[str, Any]]) -> Dict[str, float]:
    """Check that weather dims are sometimes populated (not always zero).

    If ALL samples have zero weather dims, weather encoding might be broken.
    This is a heuristic check — it's valid for weather to be absent, but it
    should appear at least sometimes in random battles.
    """
    weather_active = 0
    for s in samples:
        obs = np.asarray(s.get("obs"))
        if obs.size == 0:
            continue
        weather_dims = obs[_GLOBAL_TOKEN, :len(WEATHER_LIST)]
        if np.any(weather_dims > 0.5):
            weather_active += 1
    rate = weather_active / max(len(samples), 1)
    return {
        "obs_val/weather_active_rate": float(rate),
        "obs_val/weather_never_seen": float(weather_active == 0 and len(samples) >= 10),
    }


def _check_action_mask_binary(samples: List[Dict[str, Any]]) -> Dict[str, float]:
    """Verify action masks are binary (0.0 or 1.0) and have at least 1 valid."""
    non_binary = 0
    no_valid = 0
    valid_counts: List[int] = []
    for s in samples:
        mask = np.asarray(s.get("action_mask"))
        if mask.size == 0:
            continue
        if not np.all((mask == 0.0) | (mask == 1.0)):
            non_binary += 1
        n_valid = int(np.sum(mask > 0.5))
        valid_counts.append(n_valid)
        if n_valid == 0:
            no_valid += 1
    avg_valid = float(np.mean(valid_counts)) if valid_counts else 0.0
    return {
        "obs_val/mask_non_binary_count": float(non_binary),
        "obs_val/mask_no_valid_count": float(no_valid),
        "obs_val/mask_avg_valid_actions": avg_valid,
        "obs_val/mask_fail": float(non_binary > 0 or no_valid > 0),
    }


def _check_switch_mask_consistency(samples: List[Dict[str, Any]]) -> Dict[str, float]:
    """Check that fainted bench pokemon have their switch actions masked out.

    For each bench token that is present and fainted, the corresponding
    compressed switch action should be masked (0.0).
    """
    inconsistencies = 0
    checks_run = 0
    for s in samples:
        obs = np.asarray(s.get("obs"))
        mask = np.asarray(s.get("action_mask"))
        if obs.size == 0 or mask.size == 0:
            continue
        for k in range(5):  # 5 bench slots
            bench_token = _FIRST_BENCH_TOKEN + k
            compressed_switch = COMPRESSED_SWITCH_ACTIONS.start + k

            if bench_token >= NUM_TOKENS or compressed_switch >= COMPRESSED_ACTION_SPACE_N:
                break

            is_present = obs[bench_token, _PRESENCE_OFFSET] > 0.5
            is_fainted = obs[bench_token, _IS_FAINTED_OFFSET] > 0.5
            is_masked = mask[compressed_switch] > 0.5

            if is_present and is_fainted:
                checks_run += 1
                if is_masked:
                    inconsistencies += 1

    return {
        "obs_val/switch_fainted_inconsistencies": float(inconsistencies),
        "obs_val/switch_fainted_checks_run": float(checks_run),
        "obs_val/switch_fainted_fail": float(inconsistencies > 0),
    }


def _check_bench_fainted_flags(samples: List[Dict[str, Any]]) -> Dict[str, float]:
    """Verify fainted bench pokemon still have is_present=1 (not dropped)."""
    missing = 0
    for s in samples:
        obs = np.asarray(s.get("obs"))
        if obs.size == 0:
            continue
        # Check all bench tokens — if hp_fraction is 0, is_present should still be 1
        # (pokemon are not removed from tokens when fainted)
        for t in range(_FIRST_BENCH_TOKEN, min(_FIRST_BENCH_TOKEN + 5, NUM_TOKENS)):
            is_present = obs[t, _PRESENCE_OFFSET]
            is_fainted = obs[t, _IS_FAINTED_OFFSET]
            # If fainted flag is set but present flag is not, something is wrong
            if is_fainted > 0.5 and is_present < 0.5:
                missing += 1
    return {
        "obs_val/bench_fainted_but_not_present": float(missing),
        "obs_val/bench_fainted_fail": float(missing > 0),
    }


def _check_species_nonzero(samples: List[Dict[str, Any]]) -> Dict[str, float]:
    """Verify active pokemon tokens have non-zero species IDs."""
    zero_species = 0
    for s in samples:
        species = np.asarray(s.get("species"))
        if species.size == 0:
            continue
        # Active tokens should have species > 0
        if species[_OUR_ACTIVE_TOKEN] == 0:
            zero_species += 1
        if _OPP_ACTIVE_TOKEN < len(species) and species[_OPP_ACTIVE_TOKEN] == 0:
            zero_species += 1
    return {
        "obs_val/active_zero_species": float(zero_species),
        "obs_val/active_zero_species_fail": float(zero_species > 0),
    }
