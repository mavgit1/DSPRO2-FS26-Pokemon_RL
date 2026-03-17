import os
import time
import json
from typing import Optional, Dict, Any

from dotenv import load_dotenv
import mlflow
import torch
import gymnasium as gym

import ray
from ray.tune.registry import register_env
from ray.rllib.core.rl_module.rl_module import RLModuleSpec
from ray.rllib.algorithms.ppo import PPOConfig

# Local imports
from src.config.TM_optimal_config import (
    TrainingConfig,
    CurriculumStageConfig,
    get_config,
)
from src.envs.battle_env import create_env_creator, get_observation_space
from src.training.callbacks import CurriculumManager, CheckpointManager


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
    ):
        """
        Initialize trainer.
        
        Args:
            config: Training configuration (overrides preset)
            preset: Config preset name ("quick", "standard", "optimal", "large")
            num_servers: Number of Showdown servers
            start_port: Starting port for servers
        """
        # Load config
        self.config = config or get_config(preset)
        
        # Server settings
        self.num_servers = num_servers
        self.start_port = start_port
        
        # Initialize components
        self.curriculum = None
        if self.config.curriculum.enabled and self.config.curriculum.stages:
            self.curriculum = CurriculumManager(self.config.curriculum)
            self._validate_curriculum_config()
        
        self.checkpoint_mgr = CheckpointManager(
            self.config.checkpoint_dir,
            self.config.keep_checkpoints_num
        )
        
        # Algorithm and tracking (set during training)
        self.algo = None
        self.total_steps = 0
        self.iteration = 0
        self.best_reward = float('-inf')

    def _validate_curriculum_config(self) -> None:
        if not self.curriculum:
            return

        stages = self.config.curriculum.stages
        names = [s.name for s in stages]
        if len(set(names)) != len(names):
            raise ValueError(f"Curriculum stage names must be unique. Got: {names}")

        for stage in stages:
            if not (0.0 <= stage.promote_at_win_rate <= 1.0 or stage.promote_at_win_rate > 1.0):
                raise ValueError(
                    f"Invalid threshold for stage '{stage.name}': {stage.promote_at_win_rate}"
                )
            if stage.min_samples_for_promotion <= 0:
                raise ValueError(
                    f"min_samples_for_promotion must be > 0 for stage '{stage.name}'"
                )
            if not stage.opponent_mix:
                raise ValueError(f"opponent_mix cannot be empty for stage '{stage.name}'")
            if sum(v for v in stage.opponent_mix.values() if float(v) > 0) <= 0:
                raise ValueError(
                    f"opponent_mix must contain at least one positive weight for stage '{stage.name}'"
                )

    def _foreach_env(self, fn):
        """Run a function on every remote env instance."""
        def _call_on_worker(worker):
            if hasattr(worker, "foreach_env"):
                return worker.foreach_env(fn)
            if hasattr(worker, "env"):
                env_obj = worker.env
                return [fn(env_obj)] if env_obj is not None else []
            return []

        # Newer RLlib API: prefer env_runner_group. Some versions hard-error
        # if `workers` is touched at all.
        runner_attr = getattr(self.algo, "env_runner_group", None)
        try:
            runner_group = runner_attr() if callable(runner_attr) else runner_attr
        except (TypeError, ValueError):
            runner_group = None
        if runner_group is not None and hasattr(runner_group, "foreach_worker"):
            return runner_group.foreach_worker(_call_on_worker)

        # Backward compatibility for older RLlib only.
        workers_attr = getattr(self.algo, "workers", None)
        try:
            workers_group = workers_attr() if callable(workers_attr) else workers_attr
        except (TypeError, ValueError):
            workers_group = None
        if workers_group is not None and hasattr(workers_group, "foreach_worker"):
            return workers_group.foreach_worker(_call_on_worker)

        return []

    @staticmethod
    def _flatten_for_mlflow(prefix: str, value: Any, out: Dict[str, Any]) -> None:
        if isinstance(value, dict):
            for k, v in value.items():
                key = f"{prefix}.{k}" if prefix else str(k)
                PokemonTrainer._flatten_for_mlflow(key, v, out)
            return
        if isinstance(value, list):
            for idx, item in enumerate(value):
                key = f"{prefix}.{idx}" if prefix else str(idx)
                PokemonTrainer._flatten_for_mlflow(key, item, out)
            return

        if isinstance(value, (str, int, float, bool)):
            out[prefix] = value
        else:
            out[prefix] = json.dumps(value, default=str)

    def _collect_recent_outcomes(self) -> list[int]:
        """Collect terminal outcomes (1 win / 0 loss) from all envs."""
        nested = self._foreach_env(
            lambda e: e.pop_recent_outcomes() if hasattr(e, "pop_recent_outcomes") else []
        )
        outcomes: list[int] = []
        if not nested:
            return outcomes

        for worker_item in nested:
            if not isinstance(worker_item, list):
                continue
            for env_item in worker_item:
                if isinstance(env_item, list):
                    outcomes.extend(int(v) for v in env_item if v in {0, 1})
        return outcomes

    def _apply_curriculum_stage(self, stage: CurriculumStageConfig) -> None:
        """Push stage payload to all running env wrappers."""
        payload = stage.to_dict()
        self._foreach_env(
            lambda e: e.apply_curriculum_stage(payload)
            if hasattr(e, "apply_curriculum_stage")
            else None
        )
        print(
            f"Applied stage '{stage.name}' | threshold={stage.promote_at_win_rate:.2f} "
            f"| mix={stage.opponent_mix}"
        )
    
    def _register_environments(self) -> None:
        """Register environments with Ray."""
        initial_stage = None
        if self.curriculum:
            initial_stage = self.curriculum.current_stage

        for i in range(self.num_servers):
            port = self.start_port + i
            env_name = f"pokemon_battle_{port}"
            
            env_creator = create_env_creator(
                battle_format=self.config.env.battle_format,
                server_host=self.config.env.showdown_host,
                server_port=port,
                reward_config=(
                    initial_stage.reward_config if initial_stage else self.config.reward
                ),
                opponent_mix=(initial_stage.opponent_mix if initial_stage else None),
            )
            
            register_env(env_name, env_creator)
        
        print(f"Registered {self.num_servers} environments")
    
    def _build_config(self) -> PPOConfig:
        """Build PPO configuration."""
        # Import model here to avoid circular imports
        from src.models.battle_transformer import PokemonRLModule
        
        # Primary environment
        env_name = f"pokemon_battle_{self.start_port}"
        
        config = (
            PPOConfig()
            .environment(
                env=env_name,
                env_config={},
            )
            .framework("torch")
            # -----------------------------------------------------------------
            # ENFORCE NEW API STACK (Block the old way)
            # -----------------------------------------------------------------
            .api_stack(
                enable_rl_module_and_learner=True,
                enable_env_runner_and_connector_v2=True,
            )
            # -----------------------------------------------------------------
            # PPO HYPERPARAMETERS (Standard)
            # -----------------------------------------------------------------
            .training(
                lr=self.config.ppo.lr,
                gamma=self.config.ppo.gamma,
                lambda_=self.config.ppo.lambda_,
                clip_param=self.config.ppo.clip_param,
                entropy_coeff=self.config.ppo.entropy_coeff,
                vf_loss_coeff=self.config.ppo.vf_loss_coeff,
                vf_clip_param=self.config.ppo.vf_clip_param,
                grad_clip=self.config.ppo.grad_clip,
                train_batch_size=self.config.ppo.train_batch_size,
                minibatch_size=self.config.ppo.sgd_minibatch_size,
                num_epochs=self.config.ppo.num_sgd_iter,
            )
            # -----------------------------------------------------------------
            # MODEL
            # -----------------------------------------------------------------
            .rl_module(
                rl_module_spec=RLModuleSpec(
                    module_class=PokemonRLModule,
                    observation_space=get_observation_space(),
                    action_space=gym.spaces.Discrete(22),
                    model_config={
                        "custom_model_config": self.config.model.to_dict(),
                    },
                )
            )
            # -----------------------------------------------------------------
            # PARALLELISM
            # -----------------------------------------------------------------
            .env_runners(
                num_env_runners=self.config.env.num_workers,
                num_envs_per_env_runner=self.config.env.num_envs_per_worker,
            )
            # -----------------------------------------------------------------
            # HARDWARE
            # -----------------------------------------------------------------
            .learners(
                num_learners=torch.cuda.device_count() if torch.cuda.is_available() and torch.cuda.device_count() > 1 else 0,
                num_gpus_per_learner=1 if torch.cuda.is_available() else 0,
            )
            # -----------------------------------------------------------------
            # DEBUGGING
            # -----------------------------------------------------------------
            .debugging(log_level="WARNING")
        )
        
        return config
    
    def train_step(self) -> Dict[str, Any]:
        result = self.algo.train()
        self.total_steps = int(result.get("num_env_steps_sampled_lifetime", self.total_steps))
        self.iteration += 1
        return result
    
    def should_checkpoint(self) -> bool:
        """Check if we should save a checkpoint."""
        return self.total_steps >= (self.iteration * self.config.checkpoint_freq)
    
    def should_print(self) -> bool:
        """Check if we should print progress."""
        return self.total_steps >= (self.iteration * self.config.print_freq)
    
    def train(self) -> None:
        """Run the training loop."""
        # Initialize Ray
        ray.init(
            ignore_reinit_error=True,
            num_gpus=torch.cuda.device_count() if torch.cuda.is_available() else 0,
        )
        
        # Register environments
        self._register_environments()
        
        # Build algorithm
        print("Building algorithm...")
        ppo_config = self._build_config()
        self.algo = ppo_config.build_algo()
        
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
        with mlflow.start_run():
            # Log config parameters to MLflow
            flat_params: Dict[str, Any] = {}
            self._flatten_for_mlflow("", self.config.to_dict(), flat_params)
            mlflow.log_params(flat_params)
            
            try:
                if self.curriculum:
                    self._apply_curriculum_stage(self.curriculum.current_stage)

                while self.total_steps < self.config.total_timesteps:
                    # Train
                    result = self.train_step()
                    
                    env_stats = result.get("env_runners", {})
                    reward_mean = float(env_stats.get("episode_return_mean", 0.0)) # Note: 'return'
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

                    # Curriculum updates (training-only, binary outcomes).
                    if self.curriculum:
                        outcomes = self._collect_recent_outcomes()
                        stage_changed = self.curriculum.update(outcomes)
                        curriculum_metrics = self.curriculum.metrics()

                        rolling_win = curriculum_metrics.get("curriculum_rolling_win_rate")
                        if rolling_win is not None:
                            metrics["curriculum_rolling_win_rate"] = float(rolling_win)
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

                    mlflow.log_metrics(metrics, step=self.total_steps)
                    
                    # Print progress
                    if self.iteration % 10 == 0:
                        elapsed = (time.time() - start_time) / 3600
                        print(f"Iter {self.iteration} | Steps: {self.total_steps:,} | Reward: {reward_mean:.2f} | Time: {elapsed:.2f}h")
                    
                    # Checkpoint
                    if self.should_checkpoint():
                        ckpt_path = self.checkpoint_mgr.save_checkpoint(
                            self.algo, self.total_steps
                        )
                        print(f"Checkpoint saved: {ckpt_path}")
                    
            except KeyboardInterrupt:
                print("Training interrupted by user")
            
            finally:
                # Final checkpoint
                save_dir = os.path.abspath(f"{self.config.checkpoint_dir}/final")
                save_result = self.algo.save(save_dir)
                
                final_path = save_result.checkpoint.path
                
                mlflow.log_artifacts(local_dir=final_path, artifact_path="final_model")
                
                # Summary
                elapsed = time.time() - start_time
                print("=" * 60)
                print("Training Complete")
                print("=" * 60)
                print(f"Total Steps: {self.total_steps:,}")
                print(f"Total Time: {elapsed/3600:.1f} hours")
                print(f"Best Reward: {self.best_reward:.2f}")
                print(f"Final Model: {final_path}")
                print("=" * 60)
                
                # Cleanup
                self.algo.stop()
                ray.shutdown()


# =============================================================================
# CONVENIENCE FUNCTION
# =============================================================================

def train(
    preset: str = "standard",
    num_servers: int = 1,
    start_port: int = 8000,
    total_timesteps: Optional[int] = None,
) -> None:
    """
    Quick training function.
    
    Args:
        preset: Config preset ("quick", "standard", "optimal", "large")
        num_servers: Number of Showdown servers
        start_port: Starting port for servers
        total_timesteps: Override total timesteps
    """
    config = get_config(preset)
    
    if total_timesteps:
        config.total_timesteps = total_timesteps
    
    trainer = PokemonTrainer(
        config=config,
        num_servers=num_servers,
        start_port=start_port,
    )
    
    trainer.train()