#!/usr/bin/env python3
"""Optuna-based hyperparameter sweep for Pokemon RL PPO agent.

Usage:
    uv run scripts/hparam_sweep.py --n-trials 50
    uv run scripts/hparam_sweep.py --n-trials 50 --timesteps 500000
    uv run scripts/hparam_sweep.py --n-trials 3 --timesteps 100000   # dry run

Resume:
    Ctrl+C to stop. Re-run the same command to resume — Optuna persists
    the study to logs/hparam_study.db and skips completed trials.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import mlflow
import optuna
from dotenv import load_dotenv, find_dotenv

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config.TM_optimal_config import (
    CurriculumConfig,
    CurriculumStageConfig,
    TrainingConfig,
    get_config,
)
from src.training.trainer import PokemonTrainer


def create_trial_config(
    trial: optuna.Trial, base_preset: str, timesteps: int
) -> TrainingConfig:
    """Sample hyperparameters and build a TrainingConfig for this trial."""
    config = get_config(base_preset)
    config.total_timesteps = timesteps

    # --- PPO / Learning ---
    config.ppo.lr = trial.suggest_float("lr", 1e-4, 1e-3, log=True)
    config.ppo.entropy_coeff = trial.suggest_float("entropy_coeff", 0.01, 0.3, log=True)
    config.ppo.gamma = trial.suggest_float("gamma", 0.93, 0.99)
    config.ppo.lambda_ = trial.suggest_float("lambda_", 0.82, 0.95)
    config.ppo.clip_param = trial.suggest_float("clip_param", 0.05, 0.3)
    config.ppo.vf_clip_param = trial.suggest_float("vf_clip_param", 1.0, 5.0)
    config.model.dropout = trial.suggest_float("dropout", 0.0, 0.2)

    # --- Reward ---
    config.reward.reward_scale = trial.suggest_categorical(
        "reward_scale", [0.05, 0.1, 0.2]
    )
    config.reward.matchup_reward_weight = trial.suggest_float(
        "matchup_reward_weight", 0.0, 0.5
    )
    config.reward.action_quality_weight = trial.suggest_float(
        "action_quality_weight", 0.0, 0.5
    )

    # --- Fixed single-stage curriculum (no promotion) ---
    config.curriculum = CurriculumConfig(
        enabled=True,
        stages=[
            CurriculumStageConfig(
                name="sweep_fixed",
                promote_at_win_rate=1.01,  # never promotes
                min_samples_for_promotion=20000,
                opponent_mix={
                    "random": 0.2,
                    "random_no_switch": 0.4,
                    "heuristic": 0.4,
                },
            )
        ],
    )

    # --- Validation: run once at the end ---
    config.validation.freq_steps = timesteps
    config.validation.benchmark_episodes_per_opponent = 50
    config.validation.protocols = ["benchmark"]

    # --- Disable intermediate checkpoints ---
    config.checkpoint_freq = timesteps + 1

    return config


def objective(
    trial: optuna.Trial,
    base_preset: str,
    timesteps: int,
    num_servers: int,
    start_port: int,
    mlflow_experiment: str,
) -> float:
    """Run one training trial and return the benchmark skill score."""
    config = create_trial_config(trial, base_preset, timesteps)

    # Ensure no stale MLflow runs are active from a previous trial.
    while mlflow.active_run() is not None:
        try:
            mlflow.end_run()
        except Exception:
            break

    # Log trial params to MLflow.
    trial_params = {f"hparam/{k}": v for k, v in trial.params.items()}

    mlflow.set_experiment(mlflow_experiment)

    trainer = PokemonTrainer(
        config=config,
        preset=base_preset,
        num_servers=num_servers,
        start_port=start_port,
        mlflow_experiment_name=mlflow_experiment,
    )

    trial_start = time.time()
    print(f"\n{'=' * 60}")
    print(f"TRIAL {trial.number}")
    print(f"Params: {json.dumps(trial.params, indent=2)}")
    print(f"{'=' * 60}", flush=True)

    try:
        final_metrics = trainer.train()
    except Exception as exc:
        print(f"[ERROR] Trial {trial.number} failed: {exc}")
        raise optuna.TrialPruned() from exc
    finally:
        # Ensure the MLflow run opened by trainer.train() is closed before
        # the next trial tries to start a new one.
        try:
            mlflow.end_run()
        except Exception:
            pass

    elapsed = time.time() - trial_start
    skill_score = final_metrics.get("benchmark/skill_score", 0.0)

    # Log to MLflow in the current trial run.
    try:
        trial_params["hparam/skill_score"] = skill_score
        trial_params["hparam/elapsed_sec"] = elapsed
        trial_params.update(final_metrics)
        mlflow.log_metrics(trial_params)
    except Exception:
        pass

    print(
        f"\nTRIAL {trial.number} DONE | "
        f"skill_score={skill_score:.3f} | "
        f"elapsed={elapsed / 60:.1f}min",
        flush=True,
    )

    return skill_score


def main() -> int:
    load_dotenv(find_dotenv())

    parser = argparse.ArgumentParser(
        description="Optuna hyperparameter sweep for Pokemon RL PPO agent.",
    )
    parser.add_argument(
        "--n-trials", type=int, default=50, help="Number of Optuna trials."
    )
    parser.add_argument(
        "--timesteps", type=int, default=500_000, help="Training steps per trial."
    )
    parser.add_argument(
        "--preset",
        choices=["quick", "standard", "memory_safe", "optimal", "large"],
        default="standard",
        help="Base config preset to override.",
    )
    parser.add_argument("--num-servers", type=int, default=8, help="Showdown servers.")
    parser.add_argument(
        "--start-port", type=int, default=8000, help="Showdown start port."
    )
    parser.add_argument(
        "--study-name", type=str, default="ppo_sweep", help="Optuna study name."
    )
    parser.add_argument(
        "--mlflow-experiment",
        type=str,
        default="Pokemon_RL_HP_Tuning",
        help="MLflow experiment for tuning runs.",
    )
    args = parser.parse_args()

    db_path = Path("logs/hparam_study.db")
    db_path.parent.mkdir(parents=True, exist_ok=True)
    storage = f"sqlite:///{db_path}"

    study = optuna.create_study(
        study_name=args.study_name,
        storage=storage,
        direction="maximize",
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=42),
    )

    # Filter out already-completed trials.
    completed = len(
        [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    )
    remaining = args.n_trials - completed
    if remaining <= 0:
        print(f"Study already has {completed} completed trials. Nothing to do.")
    else:
        print(f"Study: {args.study_name} | {completed} done, {remaining} remaining")
        print(f"Storage: {db_path}")
        print(f"Timesteps/trial: {args.timesteps:,} | Base preset: {args.preset}")
        print(f"MLflow experiment: {args.mlflow_experiment}")

        study.optimize(
            lambda trial: objective(
                trial,
                base_preset=args.preset,
                timesteps=args.timesteps,
                num_servers=args.num_servers,
                start_port=args.start_port,
                mlflow_experiment=args.mlflow_experiment,
            ),
            n_trials=remaining,
        )

    # Print results.
    print("\n" + "=" * 60)
    print("SWEEP RESULTS")
    print("=" * 60)

    completed_trials = [
        t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE
    ]
    completed_trials.sort(key=lambda t: t.value, reverse=True)

    for rank, trial in enumerate(completed_trials[:10], 1):
        print(f"\n#{rank} | skill_score={trial.value:.4f} | params:")
        for k, v in trial.params.items():
            print(f"  {k}: {v}")

    if completed_trials:
        best = completed_trials[0]
        print("\n" + "=" * 60)
        print("BEST CONFIG (ready to paste into TM_optimal_config.py)")
        print("=" * 60)
        print(f"# skill_score = {best.value:.4f}")
        for k, v in best.params.items():
            if isinstance(v, float):
                print(f"{k} = {v}")
            else:
                print(f"{k} = {v!r}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
