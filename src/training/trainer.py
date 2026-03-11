import os
import sys
import time
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List

from dotenv import load_dotenv

# Load .env from project root
_project_root = Path(__file__).resolve().parent.parent.parent
load_dotenv(_project_root / '.env')

import ray
from ray.tune.registry import register_env
from ray.rllib.core.rl_module.rl_module import RLModuleSpec
from ray.rllib.algorithms.ppo import PPOConfig

# Local imports
from src.config.TM_optimal_config import TrainingConfig, get_config
from src.envs.battle_env import create_env_creator
from src.training.callbacks import TrainingLogger, CurriculumManager, CheckpointManager

logger = logging.getLogger(__name__)


class PokemonTrainer:
    """
    Main trainer class for Pokemon RL.
    
    Handles:
    - Environment registration
    - Algorithm configuration
    - Training loop with logging
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
        self.logger_util = TrainingLogger(self.config.log_dir)
        self.curriculum = CurriculumManager(
            stages=self.config.curriculum_stages
        ) if self.config.use_curriculum else None
        self.checkpoint_mgr = CheckpointManager(
            self.config.checkpoint_dir,
            self.config.keep_checkpoints_num
        )
        
        # Algorithm (set during training)
        self.algo = None
        self.total_steps = 0
        self.iteration = 0
    
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
        
        logger.info(f"Registered {self.num_servers} environments")
    
    def _build_config(self) -> PPOConfig:
        """Build PPO configuration."""
        # Import model here to avoid circular imports
        from models.battle_transformer import PokemonRLModule
        
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
                num_learners=0,
                num_gpus_per_learner=int(os.environ.get("RLLIB_NUM_GPUS", str(int(self.config.num_gpus)))),
            )
            # -----------------------------------------------------------------
            # DEBUGGING
            # -----------------------------------------------------------------
            .debugging(log_level="WARNING")
        )
        
        return config
    
    def train_step(self) -> Dict[str, Any]:
        """Execute one training iteration."""
        result = self.algo.train()
        
        self.total_steps = result.get("num_env_steps_trained", 0)
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
            num_gpus=int(self.config.num_gpus),
        )
        
        # Register environments
        self._register_environments()
        
        # Build algorithm
        logger.info("Building algorithm...")
        ppo_config = self._build_config()
        self.algo = ppo_config.build_algo()
        
        logger.info("=" * 60)
        logger.info("Starting Training")
        logger.info("=" * 60)
        logger.info(f"Battle Format: {self.config.env.battle_format}")
        logger.info(f"Total Timesteps: {self.config.total_timesteps:,}")
        logger.info(f"Workers: {self.config.env.num_workers}")
        logger.info(f"Batch Size: {self.config.ppo.train_batch_size:,}")
        logger.info(f"Model Hidden: {self.config.model.hidden_dim}")
        logger.info("=" * 60)
        
        start_time = time.time()
        
        try:
            while self.total_steps < self.config.total_timesteps:
                # Train
                result = self.train_step()
                
                # Log
                metrics = {
                    "iteration": self.iteration,
                    "num_env_steps_trained": self.total_steps,
                    "episode_reward_mean": result.get("episode_reward_mean", 0),
                    "episode_len_mean": result.get("episode_len_mean", 0),
                }
                self.logger_util.log(metrics)
                
                # Print progress
                if self.iteration % 10 == 0:
                    self.logger_util.print_summary(metrics)
                
                # Checkpoint
                if self.should_checkpoint():
                    ckpt_path = self.checkpoint_mgr.save_checkpoint(
                        self.algo, self.total_steps
                    )
                    logger.info(f"Checkpoint saved: {ckpt_path}")
                
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
            logger.info("Training interrupted by user")
        
        finally:
            # Final checkpoint
            final_path = self.algo.save(os.path.abspath(f"{self.config.checkpoint_dir}/final"))
            
            # Summary
            elapsed = time.time() - start_time
            logger.info("=" * 60)
            logger.info("Training Complete")
            logger.info("=" * 60)
            logger.info(f"Total Steps: {self.total_steps:,}")
            logger.info(f"Total Time: {elapsed/3600:.1f} hours")
            logger.info(f"Best Reward: {self.logger_util.best_reward:.2f}")
            logger.info(f"Final Model: {final_path}")
            logger.info("=" * 60)
            
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