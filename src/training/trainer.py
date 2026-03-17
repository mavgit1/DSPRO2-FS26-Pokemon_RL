import os
import time
import json
import re
import subprocess
from pathlib import Path
from typing import Optional, Dict, Any, List

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
        resume_checkpoint: Optional[str] = None,
        mlflow_run_id: Optional[str] = None,
    ):
        """
        Initialize trainer.
        
        Args:
            config: Training configuration (overrides preset)
            preset: Config preset name ("quick", "standard", "optimal", "large")
            num_servers: Number of Showdown servers
            start_port: Starting port for servers
            resume_checkpoint: Optional RLlib checkpoint path to restore from
            mlflow_run_id: Optional MLflow run ID to continue logging in the same run
        """
        # Load config
        self.config = config or get_config(preset)
        
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
        
        self.checkpoint_mgr = CheckpointManager(
            self.config.checkpoint_dir,
            self.config.keep_checkpoints_num
        )
        
        # Algorithm and tracking (set during training)
        self.algo = None
        self.total_steps = 0
        self.iteration = 0
        self.best_reward = float('-inf')
        self._last_cpu_times: Optional[tuple[int, int]] = None

    @staticmethod
    def _extract_step_from_checkpoint_path(checkpoint_path: str) -> Optional[int]:
        """Best-effort parse of sampled steps from checkpoint path."""
        match = re.search(r"step_(\d+)", checkpoint_path)
        if match:
            return int(match.group(1))
        return None

    def _resolve_resume_checkpoint(self) -> Optional[str]:
        """Resolve checkpoint path for resume, including 'latest' alias."""
        if not self.resume_checkpoint:
            return None

        if self.resume_checkpoint != "latest":
            return self.resume_checkpoint

        ckpt_root = Path(self.config.checkpoint_dir).resolve()
        if not ckpt_root.exists():
            return None

        candidates = [
            p for p in ckpt_root.rglob("*")
            if p.is_dir() and p.name.startswith("checkpoint_")
        ]
        if not candidates:
            return None

        latest = max(candidates, key=lambda p: p.stat().st_mtime)
        return str(latest)

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

    def _collect_recent_episode_stats(self) -> List[Dict[str, float]]:
        """Collect per-episode stats emitted by env wrappers."""
        nested = self._foreach_env(
            lambda e: e.pop_recent_episode_stats()
            if hasattr(e, "pop_recent_episode_stats")
            else []
        )
        stats: List[Dict[str, float]] = []
        if not nested:
            return stats
        for worker_item in nested:
            if not isinstance(worker_item, list):
                continue
            for env_item in worker_item:
                if isinstance(env_item, list):
                    for item in env_item:
                        if isinstance(item, dict):
                            stats.append(item)
        return stats

    @staticmethod
    def _find_numeric_by_substring(container: Any, key_substring: str) -> Optional[float]:
        """Return first numeric value whose flattened path contains substring."""
        target = key_substring.lower()

        def _walk(obj: Any, prefix: str = "") -> Optional[float]:
            if isinstance(obj, dict):
                for key, val in obj.items():
                    child_prefix = f"{prefix}.{key}" if prefix else str(key)
                    found = _walk(val, child_prefix)
                    if found is not None:
                        return found
                return None
            if isinstance(obj, list):
                for idx, val in enumerate(obj):
                    child_prefix = f"{prefix}.{idx}" if prefix else str(idx)
                    found = _walk(val, child_prefix)
                    if found is not None:
                        return found
                return None
            if isinstance(obj, (int, float)) and target in prefix.lower():
                return float(obj)
            return None

        return _walk(container)

    @staticmethod
    def _collect_numeric_values_for_exact_keys(
        container: Any,
        keys: List[str],
    ) -> List[float]:
        """Collect numeric values for exact key matches anywhere in nested payload."""
        keys_set = {k.lower() for k in keys}
        out: List[float] = []

        def _walk(obj: Any) -> None:
            if isinstance(obj, dict):
                for key, val in obj.items():
                    if key.lower() in keys_set and isinstance(val, (int, float)):
                        out.append(float(val))
                    _walk(val)
            elif isinstance(obj, list):
                for item in obj:
                    _walk(item)

        _walk(container)
        return out

    def _collect_ppo_metrics(self, result: Dict[str, Any]) -> Dict[str, float]:
        # Restrict to learner-specific payloads first to avoid matching static config keys.
        learner_payload = result.get("learners")
        if learner_payload is None:
            learner_payload = result.get("info", {}).get("learner")
        if learner_payload is None:
            learner_payload = result

        alias_map = {
            "ppo/policy_loss": ["policy_loss", "pi_loss", "mean_policy_loss"],
            "ppo/value_loss": ["vf_loss", "value_loss", "mean_vf_loss", "critic_loss"],
            "ppo/entropy": ["entropy", "entropy_loss", "mean_entropy"],
            "ppo/kl": ["kl", "mean_kl_loss", "kl_loss"],
            "ppo/explained_variance": ["explained_variance", "vf_explained_var"],
            "ppo/clip_fraction": ["clip_frac", "clipped", "clip_fraction"],
        }
        out: Dict[str, float] = {}
        for metric_name, aliases in alias_map.items():
            values = self._collect_numeric_values_for_exact_keys(learner_payload, aliases)
            if values:
                out[metric_name] = float(sum(values) / len(values))
        return out

    @staticmethod
    def _mean(values: List[float]) -> Optional[float]:
        if not values:
            return None
        return float(sum(values) / len(values))

    def _aggregate_episode_metrics(
        self, outcomes: List[int], episode_stats: List[Dict[str, float]]
    ) -> Dict[str, float]:
        metrics: Dict[str, float] = {}

        if outcomes:
            wins = sum(1 for x in outcomes if x == 1)
            losses = sum(1 for x in outcomes if x == 0)
            total = len(outcomes)
            metrics["outcome/wins_interval"] = float(wins)
            metrics["outcome/losses_interval"] = float(losses)
            metrics["outcome/draws_or_unknown_interval"] = float(max(0, total - wins - losses))
            metrics["outcome/win_rate_interval"] = float(wins / total) if total > 0 else 0.0

        if not episode_stats:
            return metrics

        reward_victory = [float(s["reward_victory_component"]) for s in episode_stats if "reward_victory_component" in s]
        reward_hp = [float(s["reward_hp_diff_component"]) for s in episode_stats if "reward_hp_diff_component" in s]
        reward_faint = [float(s["reward_faint_component"]) for s in episode_stats if "reward_faint_component" in s]
        reward_step = [float(s["reward_step_penalty_component"]) for s in episode_stats if "reward_step_penalty_component" in s]
        our_hp = [float(s["terminal_our_hp_remaining"]) for s in episode_stats if "terminal_our_hp_remaining" in s]
        opp_hp = [float(s["terminal_opp_hp_remaining"]) for s in episode_stats if "terminal_opp_hp_remaining" in s]
        faint_diff = [float(s["terminal_faint_diff"]) for s in episode_stats if "terminal_faint_diff" in s]
        turns = [float(s["battle_turns"]) for s in episode_stats if "battle_turns" in s]
        win_turns = [
            float(s["battle_turns"])
            for s in episode_stats
            if "battle_turns" in s and float(s.get("outcome", -1.0)) == 1.0
        ]
        mask_valid = [float(s["action_mask_valid_count_mean"]) for s in episode_stats if "action_mask_valid_count_mean" in s]
        total_actions = [float(s["episode_total_actions"]) for s in episode_stats if "episode_total_actions" in s]
        attack_actions = [float(s["episode_attack_actions"]) for s in episode_stats if "episode_attack_actions" in s]
        switch_actions = [float(s["episode_switch_actions"]) for s in episode_stats if "episode_switch_actions" in s]
        fallback_events = [float(s["episode_fallback_events"]) for s in episode_stats if "episode_fallback_events" in s]

        mean_map = {
            "reward/reward_victory_component_mean": reward_victory,
            "reward/reward_hp_diff_component_mean": reward_hp,
            "reward/reward_faint_component_mean": reward_faint,
            "reward/reward_step_penalty_component_mean": reward_step,
            "battle/terminal_our_hp_remaining_mean": our_hp,
            "battle/terminal_opp_hp_remaining_mean": opp_hp,
            "battle/terminal_faint_diff_mean": faint_diff,
            "battle/episode_turns_mean": turns,
            "battle/avg_turns_to_win": win_turns,
            "action/action_mask_valid_count_mean": mask_valid,
        }
        for metric_name, vals in mean_map.items():
            mean_val = self._mean(vals)
            if mean_val is not None:
                metrics[metric_name] = mean_val

        total_action_sum = float(sum(total_actions))
        if total_action_sum > 0.0:
            metrics["action/attack_action_ratio"] = float(sum(attack_actions) / total_action_sum)
            metrics["action/switch_action_ratio"] = float(sum(switch_actions) / total_action_sum)
            metrics["action/illegal_action_fallback_rate"] = float(
                sum(fallback_events) / total_action_sum
            )

        return metrics

    def _collect_runtime_metrics(
        self,
        result: Dict[str, Any],
        train_time_ms: float,
        steps_delta: int,
        wall_delta_s: float,
    ) -> Dict[str, float]:
        metrics: Dict[str, float] = {
            "sys/train_time_ms": float(train_time_ms),
            "sys/env_steps_delta": float(max(0, steps_delta)),
        }
        if wall_delta_s > 0:
            metrics["sys/env_steps_per_sec"] = float(max(0, steps_delta) / wall_delta_s)

        sample_time = self._find_numeric_by_substring(result, "sample_time_ms")
        learner_time = self._find_numeric_by_substring(result, "learner_update_time_ms")
        if sample_time is None:
            sample_time = self._find_numeric_by_substring(result, "sample_ms")
        if learner_time is None:
            learner_time = self._find_numeric_by_substring(result, "learn_time_ms")
        if sample_time is not None:
            metrics["sys/sample_time_ms"] = float(sample_time)
        if learner_time is not None:
            metrics["sys/learner_update_time_ms"] = float(learner_time)
        return metrics

    def _read_cpu_percent_linux(self) -> Optional[float]:
        try:
            with open("/proc/stat", "r", encoding="utf-8") as f:
                first = f.readline().strip()
            if not first.startswith("cpu "):
                return None
            values = [int(x) for x in first.split()[1:]]
            idle = values[3] + (values[4] if len(values) > 4 else 0)
            total = sum(values)
            if self._last_cpu_times is None:
                self._last_cpu_times = (idle, total)
                return None
            last_idle, last_total = self._last_cpu_times
            self._last_cpu_times = (idle, total)
            delta_total = total - last_total
            delta_idle = idle - last_idle
            if delta_total <= 0:
                return None
            busy = max(0.0, float(delta_total - delta_idle))
            return float(100.0 * (busy / float(delta_total)))
        except OSError:
            return None

    @staticmethod
    def _read_mem_metrics_linux() -> Dict[str, float]:
        out: Dict[str, float] = {}
        try:
            meminfo: Dict[str, int] = {}
            with open("/proc/meminfo", "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.split(":")
                    if len(parts) != 2:
                        continue
                    key = parts[0].strip()
                    value_part = parts[1].strip().split()[0]
                    meminfo[key] = int(value_part)
            mem_total = meminfo.get("MemTotal")
            mem_available = meminfo.get("MemAvailable")
            if mem_total and mem_available is not None:
                mem_used = max(0, mem_total - mem_available)
                out["sys/ram_used_gb"] = float(mem_used / 1024.0 / 1024.0)
                out["sys/ram_percent"] = float((mem_used / mem_total) * 100.0)
        except OSError:
            return out
        return out

    def _collect_gpu_metrics(self) -> Dict[str, float]:
        out: Dict[str, float] = {
            "sys/cuda_available": 1.0 if torch.cuda.is_available() else 0.0,
            "sys/gpu_count": float(torch.cuda.device_count() if torch.cuda.is_available() else 0),
        }
        if not torch.cuda.is_available():
            return out

        # Use nvidia-smi for utilization/global memory; fallback to torch memory only.
        try:
            cmd = [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ]
            proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
            if proc.returncode == 0 and proc.stdout.strip():
                util_vals: List[float] = []
                mem_pct_vals: List[float] = []
                mem_used_vals: List[float] = []
                for idx, row in enumerate(proc.stdout.strip().splitlines()):
                    parts = [p.strip() for p in row.split(",")]
                    if len(parts) != 3:
                        continue
                    util = float(parts[0])
                    mem_used = float(parts[1])
                    mem_total = float(parts[2])
                    mem_pct = (mem_used / mem_total) * 100.0 if mem_total > 0 else 0.0
                    out[f"sys/gpu{idx}_util_percent"] = util
                    out[f"sys/gpu{idx}_mem_used_mb"] = mem_used
                    out[f"sys/gpu{idx}_mem_percent"] = mem_pct
                    util_vals.append(util)
                    mem_used_vals.append(mem_used)
                    mem_pct_vals.append(mem_pct)
                if util_vals:
                    out["sys/gpu_util_percent"] = float(sum(util_vals) / len(util_vals))
                    out["sys/gpu_mem_used_mb"] = float(sum(mem_used_vals) / len(mem_used_vals))
                    out["sys/gpu_mem_percent"] = float(sum(mem_pct_vals) / len(mem_pct_vals))
        except (OSError, ValueError):
            pass

        for idx in range(torch.cuda.device_count()):
            out[f"sys/gpu{idx}_memory_allocated_mb"] = float(
                torch.cuda.memory_allocated(idx) / (1024.0 * 1024.0)
            )
            out[f"sys/gpu{idx}_memory_reserved_mb"] = float(
                torch.cuda.memory_reserved(idx) / (1024.0 * 1024.0)
            )
        return out

    def _collect_system_metrics(self) -> Dict[str, float]:
        metrics = self._collect_gpu_metrics()
        cpu_pct = self._read_cpu_percent_linux()
        if cpu_pct is not None:
            metrics["sys/cpu_percent"] = float(cpu_pct)
        metrics.update(self._read_mem_metrics_linux())
        return metrics

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
        
        resume_path = self._resolve_resume_checkpoint()
        if resume_path:
            self.algo.restore(resume_path)
            parsed_steps = self._extract_step_from_checkpoint_path(resume_path)
            if parsed_steps is not None:
                self.total_steps = parsed_steps
            print(f"Restored checkpoint: {resume_path}")
            if parsed_steps is not None:
                print(f"Resuming from approx. steps: {parsed_steps:,}")
            else:
                print("Resuming from checkpoint (step count will refresh after next iteration).")
        
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
                self._flatten_for_mlflow("", self.config.to_dict(), flat_params)
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
                    metrics.update(self._collect_ppo_metrics(result))

                    outcomes = self._collect_recent_outcomes()
                    episode_stats = self._collect_recent_episode_stats()
                    metrics.update(self._aggregate_episode_metrics(outcomes, episode_stats))

                    # Curriculum updates (training-only, binary outcomes).
                    if self.curriculum:
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

                    now = time.time()
                    metrics.update(
                        self._collect_runtime_metrics(
                            result=result,
                            train_time_ms=(now - iter_start) * 1000.0,
                            steps_delta=(self.total_steps - prev_steps),
                            wall_delta_s=max(now - prev_wall_time, 1e-6),
                        )
                    )
                    prev_steps = self.total_steps
                    prev_wall_time = now
                    metrics.update(self._collect_system_metrics())

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
    resume_checkpoint: Optional[str] = None,
    mlflow_run_id: Optional[str] = None,
) -> None:
    """
    Quick training function.
    
    Args:
        preset: Config preset ("quick", "standard", "optimal", "large")
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
        num_servers=num_servers,
        start_port=start_port,
        resume_checkpoint=resume_checkpoint,
        mlflow_run_id=mlflow_run_id,
    )
    
    trainer.train()