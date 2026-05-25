#!/usr/bin/env python3
"""
Standalone observation pipeline validator.

Runs a handful of real poke-env random battles (not training, just env steps)
and validates the observation pipeline end-to-end. Uses the same env creator
as training so it tests the real code path.

Requirements:
    - Pokemon Showdown servers running on the configured port

Usage:
    uv run scripts/debug/validate_pipeline.py [--port 8000] [--num-battles 5]
"""

from __future__ import annotations

# ruff: noqa: E402
import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np

from src.envs.battle_env import create_env_creator
from src.action_space import (
    COMPRESSED_ACTION_SPACE_N,
    COMPRESSED_SWITCH_ACTIONS,
)
from src.models.embedding import (
    NUM_TOKENS,
)
from src.debug.observation_validator import validate_observations


def run_validation_battles(
    port: int = 8000,
    num_battles: int = 5,
) -> Dict[str, Any]:
    """Run validation battles and return results."""
    print(f"Creating env on port {port}...")
    creator = create_env_creator(
        battle_format="gen8randombattle",
        server_port=port,
        opponent_difficulty="random",
    )
    env = creator({})

    all_samples: List[Dict[str, Any]] = []
    battle_results: List[Dict[str, Any]] = []

    try:
        for battle_idx in range(num_battles):
            print(f"\n--- Battle {battle_idx + 1}/{num_battles} ---")
            obs, info = env.reset()
            done = False
            step = 0
            samples_this_battle = 0

            while not done and step < 500:
                # Validate this observation
                if isinstance(obs, dict):
                    all_samples.append(obs)
                    samples_this_battle += 1

                # Pick a random valid action from the mask
                mask = np.asarray(obs.get("action_mask", np.zeros(COMPRESSED_ACTION_SPACE_N)))
                valid_actions = np.where(mask > 0.5)[0]
                if len(valid_actions) == 0:
                    print(f"  Step {step}: NO valid actions! Breaking.")
                    break

                action = int(np.random.choice(valid_actions))

                # Detailed check on switch alignment every 10 steps
                if step % 10 == 0 and step > 0:
                    _check_switch_alignment(env, obs, action)

                result = env.step(action)
                if isinstance(result, tuple):
                    obs = result[0]
                    terminated = result[2] if len(result) >= 3 else False
                    truncated = result[3] if len(result) >= 4 else False
                    done = bool(terminated or truncated)
                else:
                    done = True
                step += 1

            print(f"  Steps: {step}, Samples: {samples_this_battle}")
            battle_results.append({"steps": step, "samples": samples_this_battle})

    except Exception as e:
        print(f"Error during battle: {e}")
        import traceback
        traceback.print_exc()
    finally:
        env.close()

    # Validate all collected samples
    print(f"\n{'='*60}")
    print(f"Validation Results ({len(all_samples)} samples)")
    print(f"{'='*60}")

    val_metrics = validate_observations(all_samples)

    for k, v in sorted(val_metrics.items()):
        marker = " *** FAIL" if k.endswith("_fail") and v > 0 else ""
        marker += " *** WARN" if k.endswith("_never_seen") and v > 0 else ""
        print(f"  {k}: {v}{marker}")

    return {"validation": val_metrics, "battles": battle_results, "total_samples": len(all_samples)}


def _check_switch_alignment(env: Any, obs: Dict[str, np.ndarray], action: int) -> None:
    """Check that switch actions correctly target the pokemon at the expected token."""
    if action not in COMPRESSED_SWITCH_ACTIONS:
        return

    obs_arr = np.asarray(obs["obs"])
    mask = np.asarray(obs.get("action_mask", np.zeros(COMPRESSED_ACTION_SPACE_N)))

    # Which bench slot does this action target?
    bench_idx = action - COMPRESSED_SWITCH_ACTIONS.start
    bench_token = 2 + bench_idx

    if bench_token >= NUM_TOKENS:
        return

    is_present = obs_arr[bench_token, 0] > 0.5
    is_fainted = obs_arr[bench_token, 2] > 0.5
    is_masked = mask[action] > 0.5 if action < len(mask) else False

    # If the pokemon at this bench token is fainted, switch should be masked out
    if is_present and is_fainted and is_masked:
        print(f"  SWITCH INCONSISTENCY: compressed={action} bench_token={bench_token} "
              f"fainted=True but masked_as_valid=True!")


def main():
    parser = argparse.ArgumentParser(description="Validate observation pipeline")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--num-battles", type=int, default=5)
    args = parser.parse_args()

    results = run_validation_battles(port=args.port, num_battles=args.num_battles)

    # Exit with error if any failures
    failures = {k: v for k, v in results["validation"].items()
                if k.endswith("_fail") and v > 0}
    if failures:
        print(f"\nFAILURES: {failures}")
        sys.exit(1)

    print("\nAll checks passed!")
    sys.exit(0)


if __name__ == "__main__":
    main()
