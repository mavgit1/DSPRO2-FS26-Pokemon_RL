"""
Self-Play Diagnostics Entry Point
==================================

Runs a short training burst with a custom opponent mix that includes self-play
(30%) so the diagnostics instrumentation built into SelfPlayPlayer can be
exercised and inspected.

Usage:
    uv run scripts/diagnose_selfplay.py                # quick preset
    uv run scripts/diagnose_selfplay.py --preset standard
    uv run scripts/diagnose_selfplay.py --timesteps 50000

Diagnostics are logged automatically every iteration:
  - logs/selfplay_diagnostics.log  (append-only text)
  - MLflow metrics (selfplay/* prefix)
"""

import argparse
import sys
from pathlib import Path

# Project root so `src.*` imports work when running from scripts/.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import mlflow  # noqa: E402
from dotenv import load_dotenv, find_dotenv  # noqa: E402

from src.config.TM_optimal_config import (  # noqa: E402
    CurriculumStageConfig,
    get_config,
)
from src.training.trainer import PokemonTrainer  # noqa: E402


def main():
    load_dotenv(find_dotenv())

    parser = argparse.ArgumentParser(
        description="Run self-play diagnostics training burst",
    )
    parser.add_argument(
        "--preset",
        type=str,
        default="quick",
        choices=["quick", "standard", "memory_safe", "optimal", "large"],
        help="Configuration preset (default: quick)",
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=None,
        help="Override total timesteps",
    )
    parser.add_argument(
        "--num-servers",
        type=int,
        default=8,
        help="Number of Showdown servers (default: 8)",
    )
    parser.add_argument(
        "--start-port",
        type=int,
        default=8000,
        help="Starting port for Showdown servers (default: 8000)",
    )
    args = parser.parse_args()

    config = get_config(args.preset)

    if args.timesteps:
        config.total_timesteps = args.timesteps

    # Override first curriculum stage to include 30% self-play.
    if config.curriculum.enabled and config.curriculum.stages:
        config.curriculum.stages[0] = CurriculumStageConfig(
            name="selfplay_diagnostics",
            promote_at_win_rate=99.0,  # never promote — stay on this stage
            min_samples_for_promotion=999_999,
            opponent_mix={"random": 0.4, "random_no_switch": 0.3, "self": 0.3},
            reward_config=config.curriculum.stages[0].reward_config,
        )

    mlflow.set_experiment("Pokemon_RL_Battler")

    print("=" * 60)
    print("Self-Play Diagnostics Run")
    print("=" * 60)
    print(f"Preset: {args.preset}")
    print(f"Total timesteps: {config.total_timesteps:,}")
    print(f"Opponent mix: {config.curriculum.stages[0].opponent_mix}")
    print("Diagnostics log: logs/selfplay_diagnostics.log")
    print("=" * 60)

    trainer = PokemonTrainer(
        config=config,
        preset=args.preset,
        num_servers=args.num_servers,
        start_port=args.start_port,
    )
    trainer.train()


if __name__ == "__main__":
    main()
