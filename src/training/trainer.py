import os
import time
from pathlib import Path
from typing import Optional, Dict, Any

from dotenv import load_dotenv
import mlflow
import torch

import ray
from ray.tune.registry import register_env
from ray.rllib.core.rl_module.rl_module import RLModuleSpec
from ray.rllib.algorithms.ppo import PPOConfig

# Local imports
from src.config.TM_optimal_config import TrainingConfig, get_config
from src.envs.battle_env import create_env_creator
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
        self.curriculum = CurriculumManager(
            stages=self.config.curriculum_stages
        ) if self.config.use_curriculum else None
        
        self.checkpoint_mgr = CheckpointManager(
            self.config.checkpoint_dir,
            self.config.keep_checkpoints_num
        )
        
        # Algorithm and tracking (set during training)
        self.algo = None
        self.total_steps = 0
        self.iteration = 0
        self.best_reward = float('-inf')
    
    def _register_environments(self) -> None:
        """Register environments with Ray."""
        for i in range(self.num_servers):
            port = self.start_port + i
            env_name = f"pokemon_battle_{port}"
            
            env_creator = create_env_creator(
                battle_format=self.config.env.battle_format,
                server_host=self.config.env.showdown_host,
                server_port=port,
                reward_config=self.config.reward,
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
                env_config={"difficulty": "easy"},
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
            mlflow.log_params({
                "preset": self.config.env.battle_format,
                "total_timesteps": self.config.total_timesteps,
                "train_batch_size": self.config.ppo.train_batch_size,
                "hidden_dim": self.config.model.hidden_dim,
                "learning_rate": self.config.ppo.lr,
                "num_workers": self.config.env.num_workers,
            })
            
            try:
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
                    
                    # Curriculum
                    if self.curriculum:
                        if self.curriculum.update(self.total_steps):
                            # Update environment difficulty
                            self.algo.workers.foreach_worker(
                                lambda w: w.foreach_env(
                                    lambda e: setattr(
                                        e, 
                                        "difficulty", 
                                        self.curriculum.current_stage
                                    ) if hasattr(e, "difficulty") else None
                                )
                            )
            
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