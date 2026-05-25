#!/usr/bin/env python3
"""
Debug training run with observation validation.

Runs a short training session using the real pipeline (poke-env random battles),
validates observations each iteration, and logs everything to LOCAL MLflow.

Requirements:
    - Pokemon Showdown servers running (./scripts/spin_up_multiple_showdown.sh)
    - Local MLflow server:  uv run mlflow server --host 0.0.0.0 --port 5000

Usage:
    uv run scripts/debug/run_debug_training.py

Environment variables (optional):
    MLFLOW_TRACKING_URI  — defaults to http://localhost:5000
    OBS_VAL_FREQ         — validate every N iterations (default: 1)
"""

from __future__ import annotations

# ruff: noqa: E402
import os
import sys
from pathlib import Path
from typing import Dict

# Force local MLflow before anything else imports mlflow
os.environ.setdefault("MLFLOW_TRACKING_URI", "http://localhost:5000")

import mlflow
from dotenv import load_dotenv, find_dotenv

# Add project root to path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

load_dotenv(find_dotenv())

from src.config.TM_optimal_config import TrainingConfig, EnvironmentConfig, ModelConfig, PPOConfig
from src.training.trainer import PokemonTrainer
from src.training.env_bridge import collect_recent_observation_samples
from src.debug.observation_validator import validate_observations


# Smaller-than-quick preset for fast iteration
DEBUG_CONFIG = TrainingConfig(
    total_timesteps=50_000,
    env=EnvironmentConfig(
        num_workers=4,
        num_envs_per_worker=2,
    ),
    model=ModelConfig(
        hidden_dim=64,
        num_heads=2,
        num_transformer_layers=1,
        use_lstm=False,
    ),
    ppo=PPOConfig(
        train_batch_size=2048,
    ),
)


class DebugPokemonTrainer(PokemonTrainer):
    """Trainer subclass that adds observation validation each iteration."""

    def __init__(self, *args, val_freq: int = 1, **kwargs):
        super().__init__(*args, **kwargs)
        self.val_freq = val_freq
        self._val_iteration = 0

    def _record_decision_diagnostics(self) -> Dict[str, float]:
        metrics = super()._record_decision_diagnostics()

        self._val_iteration += 1
        if self._val_iteration % self.val_freq != 0:
            return metrics

        # Collect fresh observation samples and validate
        raw_samples = collect_recent_observation_samples(
            self.algo, max_samples_per_env=10
        )
        if raw_samples:
            val_metrics = validate_observations(raw_samples)
            metrics.update(val_metrics)

            # Print summary of any failures
            failures = {k: v for k, v in val_metrics.items() if k.endswith("_fail") and v > 0}
            if failures:
                print(f"  [OBS VAL] FAILURES at step {self.total_steps}: {failures}")

        return metrics


def main():
    tracking_uri = os.environ["MLFLOW_TRACKING_URI"]
    print("=" * 60)
    print("Debug Training Run")
    print("=" * 60)
    print(f"MLflow tracking URI: {tracking_uri}")
    print(f"Timesteps: {DEBUG_CONFIG.total_timesteps:,}")
    print(f"Workers: {DEBUG_CONFIG.env.num_workers}")
    print(f"Batch size: {DEBUG_CONFIG.ppo.train_batch_size:,}")
    print("=" * 60)

    # Verify MLflow is reachable
    try:
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment("debug_observation_pipeline")
        print("MLflow connection OK")
    except Exception as e:
        print(f"WARNING: Cannot reach MLflow at {tracking_uri}: {e}")
        print("Start it with: uv run mlflow server --host 0.0.0.0 --port 5000")
        sys.exit(1)

    val_freq = int(os.environ.get("OBS_VAL_FREQ", "1"))

    trainer = DebugPokemonTrainer(
        config=DEBUG_CONFIG,
        preset="quick",
        num_servers=int(os.environ.get("NUM_SERVERS", "8")),
        start_port=int(os.environ.get("START_PORT", "8000")),
        val_freq=val_freq,
    )
    trainer.train()


if __name__ == "__main__":
    main()
