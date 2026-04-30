import json
import os
import random
import subprocess
import sys
import time
from collections import deque
from pathlib import Path
from typing import Any, Dict, Optional

import mlflow
import numpy as np
import ray
import torch

from src.config.TM_optimal_config import (
    CurriculumStageConfig,
    TrainingConfig,
    get_config,
)
from src.training.checkpointing import CheckpointManager
from src.training.curriculum import CurriculumManager
from src.training.env_bridge import (
    apply_curriculum_stage,
    collect_env_memory_sentinels,
    collect_recent_observation_samples,
    collect_recent_episode_stats,
    collect_recent_outcomes,
)
from src.training.metrics import (
    aggregate_episode_metrics,
    collect_ppo_metrics,
    collect_runtime_metrics,
    flatten_for_mlflow,
)
from src.training.monitoring import SystemMetricsCollector
from src.training.resume import (
    extract_step_from_checkpoint_path,
    resolve_resume_checkpoint,
)
from src.training.rllib_config_builder import build_ppo_config, register_environments


class PokemonTrainer:
    """
    Main trainer class for Pokemon RL.

    Handles:
    - Environment registration
    - Algorithm configuration
    - Training loop with MLflow logging
    - Checkpointing
    - Curriculum learning
    """

    def __init__(
        self,
        config: Optional[TrainingConfig] = None,
        preset: str = "standard",
        num_servers: int = 1,
        start_port: int = 8000,
        resume_checkpoint: Optional[str] = None,
        mlflow_run_id: Optional[str] = None,
    ):
        """
        Initialize trainer.

        Args:
            config: Training configuration (overrides preset)
            preset: Config preset name ("quick", "standard", "memory_safe", "optimal", "large")
            num_servers: Number of Showdown servers
            start_port: Starting port for servers
            resume_checkpoint: Optional RLlib checkpoint path to restore from
            mlflow_run_id: Optional MLflow run ID to continue logging in the same run
        """
        # Load config
        self.config = config or get_config(preset)
        self.preset = preset

        # Server settings
        self.num_servers = num_servers
        self.start_port = start_port
        self.resume_checkpoint = resume_checkpoint
        self.mlflow_run_id = mlflow_run_id

        # Initialize components
        self.curriculum = None
        if self.config.curriculum.enabled and self.config.curriculum.stages:
            self.curriculum = CurriculumManager(self.config.curriculum)
            self._validate_curriculum_config()

        self._win_rate_window: deque[int] = deque(
            maxlen=self.config.curriculum.rolling_window_episodes
        )

        self.checkpoint_mgr = CheckpointManager(
            self.config.checkpoint_dir, self.config.keep_checkpoints_num
        )

        # Algorithm and tracking (set during training)
        self.algo = None
        self.total_steps = 0
        self._last_checkpoint_step = 0
        self._last_validation_step = 0
        self.iteration = 0
        self.best_reward = float("-inf")
        self.system_metrics = SystemMetricsCollector()
        self.diag_samples_per_iteration = int(
            os.environ.get("DIAG_SAMPLES_PER_ITER", "3")
        )
        self.diag_max_saved_samples = int(
            os.environ.get("DIAG_MAX_SAVED_SAMPLES", "300")
        )
        self.diag_output_path = Path(
            "logs/validation/decision_diagnostics_samples.json"
        )
        self._diag_samples_saved = 0
        self._diag_pruned_total = 0
        self._diag_records: list[Dict[str, Any]] = []

    # Main training loop.
    def train(self) -> None:
        """Run the training loop."""
        seed = 42
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        ray_tmp = os.environ.get("RAY_TMPDIR", "").strip()
        ray_kwargs: Dict[str, Any] = {
            "ignore_reinit_error": True,
            "num_gpus": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        }
        if ray_tmp:
            ray_tmp_abs = os.path.abspath(os.path.expanduser(ray_tmp))
            os.makedirs(ray_tmp_abs, exist_ok=True)
            ray_kwargs["_temp_dir"] = ray_tmp_abs

        ray.init(**ray_kwargs)

        # Register environments
        self._register_environments()

        # Build algorithm
        print("Building algorithm...")
        ppo_config = self._build_config()
        self.algo = ppo_config.build_algo()

        resume_path = resolve_resume_checkpoint(
            resume_checkpoint=self.resume_checkpoint,
            checkpoint_dir=self.config.checkpoint_dir,
        )
        if resume_path:
            self.algo.restore(resume_path)
            parsed_steps = extract_step_from_checkpoint_path(resume_path)
            if parsed_steps is not None:
                self.total_steps = parsed_steps
                self._last_checkpoint_step = parsed_steps
                self._last_validation_step = parsed_steps
            print(f"Restored checkpoint: {resume_path}")
            if parsed_steps is not None:
                print(f"Resuming from approx. steps: {parsed_steps:,}")
            else:
                print(
                    "Resuming from checkpoint (step count will refresh after next iteration)."
                )

        print("=" * 60)
        print("Starting Training")
        print("=" * 60)
        print(f"Battle Format: {self.config.env.battle_format}")
        print(f"Total Timesteps: {self.config.total_timesteps:,}")
        print(f"Workers: {self.config.env.num_workers}")
        print(f"Batch Size: {self.config.ppo.train_batch_size:,}")
        print(f"Model Hidden: {self.config.model.hidden_dim}")
        print("=" * 60)

        start_time = time.time()

        # Start MLflow run
        with mlflow.start_run(run_id=self.mlflow_run_id):
            current_run = mlflow.active_run()
            if current_run is not None:
                print(f"MLflow run id: {current_run.info.run_id}")

            # For resumed MLflow runs, avoid re-logging params that may already exist.
            if not self.mlflow_run_id:
                flat_params: Dict[str, Any] = {}
                flatten_for_mlflow("", self.config.to_dict(), flat_params)
                mlflow.log_params(flat_params)
            else:
                mlflow.set_tag("resumed", "true")
                if resume_path:
                    mlflow.set_tag("resumed_from_checkpoint", resume_path)

            try:
                if self.curriculum:
                    self._apply_curriculum_stage(self.curriculum.current_stage)

                prev_steps = self.total_steps
                prev_wall_time = time.time()

                while self.total_steps < self.config.total_timesteps:
                    iter_start = time.time()
                    # Train
                    result = self.train_step()

                    env_stats = result.get("env_runners", {})
                    reward_mean = float(
                        env_stats.get("episode_return_mean", 0.0)
                    )  # Note: 'return'
                    len_mean = float(env_stats.get("episode_len_mean", 0.0))

                    # Handle None values specifically (Ray sets None before episode 1 finishes)
                    reward_mean = float(reward_mean) if reward_mean is not None else 0.0
                    len_mean = float(len_mean) if len_mean is not None else 0.0

                    self.best_reward = max(self.best_reward, reward_mean)

                    # Log metrics to MLflow
                    metrics = {
                        "episode_reward_mean": reward_mean,
                        "episode_len_mean": len_mean,
                        "iteration": self.iteration,
                    }
                    metrics.update(collect_ppo_metrics(result))

                    outcomes = collect_recent_outcomes(self.algo)
                    episode_stats = collect_recent_episode_stats(self.algo)
                    metrics.update(aggregate_episode_metrics(outcomes, episode_stats))

                    # Curriculum updates (training-only, binary outcomes).
                    if self.curriculum:
                        stage_changed = self.curriculum.update(outcomes)
                        curriculum_metrics = self.curriculum.metrics()

                        rolling_win = curriculum_metrics.get(
                            "curriculum_rolling_win_rate"
                        )
                        if rolling_win is not None:
                            w = float(rolling_win)
                            metrics["curriculum_rolling_win_rate"] = w
                            metrics["win_rate"] = w
                        elif outcomes:
                            wins = sum(1 for o in outcomes if o == 1)
                            metrics["win_rate"] = float(wins / len(outcomes))
                        metrics["curriculum_stage_idx"] = float(
                            curriculum_metrics["curriculum_stage_idx"]
                        )
                        metrics["curriculum_valid_window_samples"] = float(
                            curriculum_metrics["curriculum_valid_window_samples"]
                        )
                        metrics["curriculum_episodes_in_stage"] = float(
                            curriculum_metrics["curriculum_episodes_in_stage"]
                        )
                        metrics["curriculum_total_episodes"] = float(
                            curriculum_metrics["curriculum_total_episodes"]
                        )

                        if stage_changed:
                            self._apply_curriculum_stage(self.curriculum.current_stage)
                            mlflow.set_tag(
                                "last_curriculum_transition",
                                f"iter_{self.iteration}_{self.curriculum.current_stage.name}",
                            )
                    else:
                        for outcome in outcomes:
                            if outcome in {0, 1}:
                                self._win_rate_window.append(int(outcome))
                        if self._win_rate_window:
                            metrics["win_rate"] = float(
                                sum(self._win_rate_window) / len(self._win_rate_window)
                            )

                    now = time.time()
                    metrics.update(
                        collect_runtime_metrics(
                            result=result,
                            train_time_ms=(now - iter_start) * 1000.0,
                            steps_delta=(self.total_steps - prev_steps),
                            wall_delta_s=max(now - prev_wall_time, 1e-6),
                        )
                    )
                    prev_steps = self.total_steps
                    prev_wall_time = now
                    metrics.update(self.system_metrics.collect())
                    metrics.update(collect_env_memory_sentinels(self.algo))
                    metrics.update(self._record_decision_diagnostics())

                    mlflow.log_metrics(metrics, step=self.total_steps)

                    # Print progress
                    if self.iteration % 10 == 0:
                        elapsed = (time.time() - start_time) / 3600
                        print(
                            f"Iter {self.iteration} | Steps: {self.total_steps:,} | Reward: {reward_mean:.2f} | Time: {elapsed:.2f}h"
                        )

                    # Scheduled validation saves a checkpoint first, then evaluates it.
                    if self.should_validate():
                        ckpt_path = self._save_checkpoint()
                        self._run_scheduled_validation(
                            checkpoint_path=ckpt_path,
                            mlflow_run_id=current_run.info.run_id
                            if current_run
                            else None,
                        )
                    elif self.should_checkpoint():
                        self._save_checkpoint()

            except KeyboardInterrupt:
                print("Training interrupted by user")

            finally:
                # Final checkpoint
                final_path = None
                if self.algo is not None:
                    save_dir = os.path.abspath(f"{self.config.checkpoint_dir}/final")
                    save_result = self.algo.save(save_dir)
                    final_path = save_result.checkpoint.path
                    mlflow.log_artifacts(
                        local_dir=final_path, artifact_path="final_model"
                    )

                elapsed = time.time() - start_time
                print("=" * 60)
                print("Training Complete")
                print("=" * 60)
                print(f"Total Steps: {self.total_steps:,}")
                print(f"Total Time: {elapsed / 3600:.1f} hours")
                print(f"Best Reward: {self.best_reward:.2f}")
                if final_path:
                    print(f"Final Model: {final_path}")
                print("=" * 60)

                if self.algo is not None:
                    self.algo.stop()
                ray.shutdown()
    
    def _validate_curriculum_config(self) -> None:
        if not self.curriculum:
            return

        stages = self.config.curriculum.stages
        names = [s.name for s in stages]
        if len(set(names)) != len(names):
            raise ValueError(f"Curriculum stage names must be unique. Got: {names}")

        for stage in stages:
            if not (
                0.0 <= stage.promote_at_win_rate <= 1.0
                or stage.promote_at_win_rate > 1.0
            ):
                raise ValueError(
                    f"Invalid threshold for stage '{stage.name}': {stage.promote_at_win_rate}"
                )
            if stage.min_samples_for_promotion <= 0:
                raise ValueError(
                    f"min_samples_for_promotion must be > 0 for stage '{stage.name}'"
                )
            if not stage.opponent_mix:
                raise ValueError(
                    f"opponent_mix cannot be empty for stage '{stage.name}'"
                )
            if sum(v for v in stage.opponent_mix.values() if float(v) > 0) <= 0:
                raise ValueError(
                    f"opponent_mix must contain at least one positive weight for stage '{stage.name}'"
                )

    def _apply_curriculum_stage(self, stage: CurriculumStageConfig) -> None:
        """Push stage payload to all running env wrappers."""
        if self.algo is None:
            return
        apply_curriculum_stage(self.algo, stage)
        print(
            f"Applied stage '{stage.name}' | threshold={stage.promote_at_win_rate:.2f} "
            f"| mix={stage.opponent_mix}"
        )

    def _register_environments(self) -> None:
        """Register environments with Ray."""
        initial_stage = None
        if self.curriculum:
            initial_stage = self.curriculum.current_stage

        register_environments(
            config=self.config,
            num_servers=self.num_servers,
            start_port=self.start_port,
            initial_stage=initial_stage,
        )
        print(
            f"Env maps to Showdown ports {self.start_port}–{self.start_port + self.num_servers - 1} "
            "(deterministic: RLlib worker_index × envs_per_runner + sub-env index, mod num_servers)"
        )

    def _build_config(self):
        """Build PPO configuration."""
        return build_ppo_config(
            config=self.config,
            start_port=self.start_port,
            num_servers=self.num_servers,
        )

    def train_step(self) -> Dict[str, Any]:
        result = self.algo.train()
        self.total_steps = int(
            result.get("num_env_steps_sampled_lifetime", self.total_steps)
        )
        self.iteration += 1
        return result

    def _get_diagnostic_analyzer(self):
        if self.algo is None:
            return None
        module = None
        get_module = getattr(self.algo, "get_module", None)
        if callable(get_module):
            try:
                module = get_module("default_policy")
            except Exception:
                try:
                    module = get_module()
                except Exception:
                    module = None

        if module is not None:
            analyze_fn = getattr(module, "analyze_observation", None)
            if callable(analyze_fn):
                return analyze_fn

        get_policy = getattr(self.algo, "get_policy", None)
        if callable(get_policy):
            try:
                policy = get_policy()
                model = getattr(policy, "model", None)
                analyze_fn = getattr(model, "analyze_observation", None)
                if callable(analyze_fn):
                    return analyze_fn
            except Exception:
                return None
        return None

    # Converts obs to expected tensor shapes for the model.
    @staticmethod
    def _to_batched_obs(obs_sample: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        obs = torch.as_tensor(obs_sample["obs"], dtype=torch.float32)
        species = torch.as_tensor(obs_sample["species"], dtype=torch.long)
        items = torch.as_tensor(obs_sample["items"], dtype=torch.long)
        abilities = torch.as_tensor(obs_sample["abilities"], dtype=torch.long)
        action_mask = torch.as_tensor(obs_sample["action_mask"], dtype=torch.float32)

        if obs.dim() == 2:
            obs = obs.unsqueeze(0)
        if species.dim() == 1:
            species = species.unsqueeze(0)
        if items.dim() == 1:
            items = items.unsqueeze(0)
        if abilities.dim() == 1:
            abilities = abilities.unsqueeze(0)
        if action_mask.dim() == 1:
            action_mask = action_mask.unsqueeze(0)

        return {
            "obs": obs,
            "species": species,
            "items": items,
            "abilities": abilities,
            "action_mask": action_mask,
        }

    def _record_decision_diagnostics(self) -> Dict[str, float]:
        metrics: Dict[str, float] = {
            "diag/samples_collected_iteration": 0.0,
            "diag/samples_saved_total": float(self._diag_samples_saved),
            "diag/samples_pruned_total": float(self._diag_pruned_total),
        }
        analyze_fn = self._get_diagnostic_analyzer()
        if analyze_fn is None:
            return metrics

        raw_samples = collect_recent_observation_samples(
            self.algo, max_samples_per_env=self.diag_samples_per_iteration
        )
        if not raw_samples:
            return metrics

        selected = raw_samples[: self.diag_samples_per_iteration]
        new_records: list[Dict[str, Any]] = []
        for sample in selected:
            try:
                diag = analyze_fn(self._to_batched_obs(sample), top_k=3)
            except Exception:
                continue
            new_records.append(
                {
                    "iteration": self.iteration,
                    "total_steps": self.total_steps,
                    "diagnostics": diag,
                }
            )

        if not new_records:
            return metrics

        self._diag_records.extend(new_records)
        if len(self._diag_records) > self.diag_max_saved_samples:
            overflow = len(self._diag_records) - self.diag_max_saved_samples
            self._diag_records = self._diag_records[overflow:]
            self._diag_pruned_total += overflow

        self._diag_samples_saved = len(self._diag_records)
        self.diag_output_path.parent.mkdir(parents=True, exist_ok=True)
        self.diag_output_path.write_text(
            json.dumps(
                {
                    "meta": {
                        "max_saved_samples": self.diag_max_saved_samples,
                        "samples_per_iteration": self.diag_samples_per_iteration,
                        "pruned_total": self._diag_pruned_total,
                    },
                    "samples": self._diag_records,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

        metrics["diag/samples_collected_iteration"] = float(len(new_records))
        metrics["diag/samples_saved_total"] = float(self._diag_samples_saved)
        metrics["diag/samples_pruned_total"] = float(self._diag_pruned_total)
        return metrics

    def should_checkpoint(self) -> bool:
        """Check if we should save a checkpoint."""
        return (
            self.config.checkpoint_freq > 0
            and self.total_steps - self._last_checkpoint_step
            >= self.config.checkpoint_freq
        )

    def should_validate(self) -> bool:
        """Check if scheduled checkpoint validation should run."""
        validation = self.config.validation
        return (
            validation.enabled
            and validation.freq_steps > 0
            and bool(validation.protocols)
            and self.total_steps - self._last_validation_step >= validation.freq_steps
        )

    def _save_checkpoint(self) -> Path:
        ckpt_path = self.checkpoint_mgr.save_checkpoint(self.algo, self.total_steps)
        self._last_checkpoint_step = self.total_steps
        print(f"Checkpoint saved: {ckpt_path}")
        return ckpt_path

    def _manifest_for_validation_protocol(self, protocol: str) -> str | None:
        if protocol == "fixed_paired":
            return self.config.validation.fixed_pair_manifest
        if protocol == "mirror":
            return self.config.validation.mirror_manifest
        return None

    # Just calls the validate_checkpoint.py script. Use the script directly if you want manual validation.
    def _run_scheduled_validation(
        self, checkpoint_path: Path, mlflow_run_id: str | None
    ) -> None:
        """Run configured validation protocols against a saved checkpoint."""
        validation = self.config.validation
        output_dir = Path("logs/validation") / f"step_{self.total_steps}"
        output_dir.mkdir(parents=True, exist_ok=True)

        for protocol in validation.protocols:
            command = [
                sys.executable,
                "scripts/validate_checkpoint.py",
                "--checkpoint",
                str(checkpoint_path),
                "--protocol",
                protocol,
                "--preset",
                self.preset,
                "--num-servers",
                str(validation.num_servers),
                "--start-port",
                str(self.start_port),
                "--max-steps-per-battle",
                str(validation.max_steps_per_battle),
                "--seed",
                str(validation.seed),
                "--output-json",
                str(output_dir / f"{protocol}_validation_report.json"),
            ]

            team_manifest = self._manifest_for_validation_protocol(protocol)
            if team_manifest:
                command.extend(["--team-manifest", team_manifest])

            if self.config.model.use_lstm:
                command.append("--use-lstm")

            if mlflow_run_id:
                command.extend(
                    [
                        "--mlflow",
                        "--mlflow-run-id",
                        mlflow_run_id,
                        "--mlflow-step",
                        str(self.total_steps),
                        "--metric-prefix",
                        f"validation/{protocol}",
                        "--experiment-name",
                        "Pokemon_RL_Battler",
                    ]
                )

            print(
                f"Running scheduled validation '{protocol}' at step {self.total_steps:,}",
                flush=True,
            )
            try:
                subprocess.run(command, check=True)
            except subprocess.CalledProcessError as exc:
                message = (
                    f"Scheduled validation '{protocol}' failed with exit code "
                    f"{exc.returncode}."
                )
                if validation.continue_on_failure:
                    print(f"{message} Continuing training.")
                    continue
                raise RuntimeError(message) from exc

        self._last_validation_step = self.total_steps

    def should_print(self) -> bool:
        """Check if we should print progress."""
        return self.total_steps >= (self.iteration * self.config.print_freq)



# =============================================================================
# CONVENIENCE FUNCTION
# =============================================================================


def train(
    preset: str = "standard",
    num_servers: int = 1,
    start_port: int = 8000,
    total_timesteps: Optional[int] = None,
    resume_checkpoint: Optional[str] = None,
    mlflow_run_id: Optional[str] = None,
) -> None:
    """
    Quick training function.

    Args:
        preset: Config preset ("quick", "standard", "memory_safe", "optimal", "large")
        num_servers: Number of Showdown servers
        start_port: Starting port for servers
        total_timesteps: Override total timesteps
        resume_checkpoint: Optional RLlib checkpoint path ("latest" supported)
        mlflow_run_id: Optional MLflow run ID to resume logging
    """
    config = get_config(preset)

    if total_timesteps:
        config.total_timesteps = total_timesteps

    trainer = PokemonTrainer(
        config=config,
        preset=preset,
        num_servers=num_servers,
        start_port=start_port,
        resume_checkpoint=resume_checkpoint,
        mlflow_run_id=mlflow_run_id,
    )

    trainer.train()
