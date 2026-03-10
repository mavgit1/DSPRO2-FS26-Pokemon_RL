import json
import time
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime

import numpy as np

logger = logging.getLogger(__name__)


class TrainingLogger:
    """
    Custom training logger for tracking metrics.
    """
    
    def __init__(self, log_dir: str = "logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize log file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = self.log_dir / f"training_{timestamp}.jsonl"
        
        # Metrics tracking
        self.metrics_history: List[Dict[str, Any]] = []
        self.best_reward = float('-inf')
        self.start_time = time.time()
    
    def log(self, metrics: Dict[str, Any]) -> None:
        """Log metrics to file and history."""
        metrics["timestamp"] = time.time()
        metrics["elapsed_hours"] = (time.time() - self.start_time) / 3600
        
        self.metrics_history.append(metrics)
        
        # Update best reward
        if "episode_reward_mean" in metrics:
            self.best_reward = max(self.best_reward, metrics["episode_reward_mean"])
        
        # Write to file
        with open(self.log_file, 'a') as f:
            f.write(json.dumps(metrics) + '\n')
    
    def print_summary(self, metrics: Dict[str, Any]) -> None:
        """Print a formatted summary of current metrics."""
        elapsed = metrics.get("elapsed_hours", 0)
        total_steps = metrics.get("num_env_steps_trained", 0)
        reward_mean = metrics.get("episode_reward_mean", 0)
        episode_len = metrics.get("episode_len_mean", 0)
        
        steps_per_sec = total_steps / (elapsed * 3600) if elapsed > 0 else 0
        
        logger.info(
            f"Steps: {total_steps:,} | "
            f"Reward: {reward_mean:.2f} (best: {self.best_reward:.2f}) | "
            f"Len: {episode_len:.0f} | "
            f"Speed: {steps_per_sec:.0f} steps/h | "
            f"Time: {elapsed:.1f}h"
        )


class CurriculumManager:
    """
    Manages curriculum learning progression.
    """
    
    def __init__(
        self,
        stages: List[str],
        interval: int = 1_000_000,
    ):
        self.stages = stages
        self.interval = interval
        self.current_stage_idx = 0
    
    @property
    def current_stage(self) -> str:
        return self.stages[self.current_stage_idx]
    
    def update(self, total_steps: int) -> bool:
        """
        Check if curriculum should advance.
        
        Returns:
            True if stage changed, False otherwise
        """
        target_idx = min(
            total_steps // self.interval,
            len(self.stages) - 1
        )
        
        if target_idx > self.current_stage_idx:
            old_stage = self.current_stage
            self.current_stage_idx = target_idx
            logger.info(f"📚 Curriculum: {old_stage} → {self.current_stage}")
            return True
        
        return False
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "stages": self.stages,
            "interval": self.interval,
            "current_stage": self.current_stage,
        }


class CheckpointManager:
    """
    Manages model checkpoints.
    """
    
    def __init__(
        self,
        checkpoint_dir: str = "checkpoints",
        keep_num: int = 5,
    ):
        self.checkpoint_dir = Path(checkpoint_dir).resolve()
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.keep_num = keep_num
        self.checkpoints: List[Path] = []
    
    def save_checkpoint(self, algo, step: int) -> Path:
        """Save checkpoint and manage rotation."""
        checkpoint_path = algo.save(str(self.checkpoint_dir / f"step_{step}"))
        self.checkpoints.append(Path(checkpoint_path))
        
        # Remove old checkpoints
        while len(self.checkpoints) > self.keep_num:
            old_ckpt = self.checkpoints.pop(0)
            if old_ckpt.exists():
                import shutil
                shutil.rmtree(old_ckpt, ignore_errors=True)
        
        return checkpoint_path
    
    def load_latest(self, algo) -> bool:
        """Load latest checkpoint if available."""
        if not self.checkpoints:
            return False
        
        latest = self.checkpoints[-1]
        algo.load(str(latest))
        logger.info(f"Loaded checkpoint: {latest}")
        return True


def compute_training_stats(
    rewards: List[float],
    lengths: List[int],
    window: int = 100,
) -> Dict[str, float]:
    """Compute training statistics."""
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
