from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import ray
import torch
from ray.rllib.core.columns import Columns

from src.config.TM_optimal_config import TrainingConfig, get_config
from src.envs.battle_env import create_env_creator
from src.training.resume import resolve_resume_checkpoint
from src.training.rllib_config_builder import build_ppo_config, register_environments
from src.validation.metrics import BattleResult, aggregate_validation_metrics
from src.validation.protocols import ValidationProtocol


def build_validation_config(preset: str) -> TrainingConfig:
    """Build a lightweight config for checkpoint validation."""
    config = get_config(preset)
    config.env.num_workers = 0
    config.env.num_envs_per_worker = 1
    config.ppo.train_batch_size = min(config.ppo.train_batch_size, 512)
    config.ppo.sgd_minibatch_size = min(config.ppo.sgd_minibatch_size, 128)
    return config


def run_validation(
    protocol: ValidationProtocol,
    checkpoint: str,
    preset: str,
    num_servers: int,
    start_port: int,
    max_steps_per_battle: int,
    seed: int,
) -> Dict[str, Any]:
    """Restore a checkpoint and run a validation protocol."""
    if protocol.name != "smoke":
        raise NotImplementedError(
            f"Protocol '{protocol.name}' is planned but not implemented yet."
        )

    _seed_everything(seed)
    config = build_validation_config(preset)
    checkpoint_path = resolve_resume_checkpoint(checkpoint, config.checkpoint_dir)
    if checkpoint_path is None:
        raise FileNotFoundError(
            f"Could not resolve checkpoint '{checkpoint}' in {config.checkpoint_dir}"
        )

    ray.init(ignore_reinit_error=True, num_gpus=0)
    algo = None
    env = None
    try:
        register_environments(
            config=config,
            num_servers=num_servers,
            start_port=start_port,
            initial_stage=None,
        )
        algo = build_ppo_config(
            config=config,
            start_port=start_port,
            num_servers=num_servers,
        ).build_algo()
        algo.restore(checkpoint_path)

        env = _build_validation_env(
            config=config,
            opponent_type=protocol.opponent,
            start_port=start_port,
        )
        results = _run_episodes(
            algo=algo,
            env=env,
            protocol=protocol,
            max_steps_per_battle=max_steps_per_battle,
        )
    finally:
        if env is not None:
            env.close()
        if algo is not None:
            algo.stop()
        ray.shutdown()

    metrics = aggregate_validation_metrics(results)
    return {
        "metadata": {
            "protocol": protocol.name,
            "checkpoint": str(Path(checkpoint_path).resolve()),
            "preset": preset,
            "opponent": protocol.opponent,
            "seed": seed,
            "max_steps_per_battle": max_steps_per_battle,
        },
        "metrics": metrics,
        "episodes": [result.to_dict() for result in results],
    }


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _build_validation_env(
    config: TrainingConfig,
    opponent_type: str,
    start_port: int,
):
    env_creator = create_env_creator(
        battle_format=config.env.battle_format,
        server_host=config.env.showdown_host,
        server_port=start_port,
        reward_config=config.reward,
        opponent_difficulty=opponent_type,
        opponent_mix={opponent_type: 1.0},
    )
    return env_creator(
        {
            "server_port": start_port,
            "num_servers": 1,
            "start_port": start_port,
            "num_envs_per_worker": 1,
            "opponent_difficulty": opponent_type,
            "opponent_mix": {opponent_type: 1.0},
        }
    )


def _run_episodes(
    algo,
    env,
    protocol: ValidationProtocol,
    max_steps_per_battle: int,
) -> List[BattleResult]:
    results: List[BattleResult] = []
    for episode_idx in range(protocol.episodes):
        obs, _info = env.reset()
        total_reward = 0.0
        steps = 0
        terminated = False
        truncated = False

        while not terminated and not truncated and steps < max_steps_per_battle:
            action = _compute_action(algo, obs)
            obs, reward, terminated, truncated, _info = env.step(action)
            total_reward += float(reward)
            steps += 1

        episode_stats = _episode_stats(env)
        outcome = _episode_outcome(
            episode_stats=episode_stats,
            terminated=terminated,
            truncated=truncated,
        )
        results.append(
            BattleResult(
                episode=episode_idx,
                opponent_type=protocol.opponent,
                outcome=outcome,
                total_reward=total_reward,
                steps=steps,
                fallback_events=int(episode_stats.get("episode_fallback_events", 0)),
                attack_actions=int(episode_stats.get("episode_attack_actions", 0)),
                switch_actions=int(episode_stats.get("episode_switch_actions", 0)),
            )
        )

    return results


def _compute_action(algo, obs: Dict[str, Any]) -> int:
    compute_single_action = getattr(algo, "compute_single_action", None)
    if callable(compute_single_action):
        try:
            action = compute_single_action(obs, explore=False)
            if isinstance(action, tuple):
                action = action[0]
            return int(action)
        except Exception:
            pass

    module = _get_module(algo)
    batch = {Columns.OBS: _to_batched_tensors(obs)}
    with torch.no_grad():
        forward = getattr(module, "forward_inference", None)
        if callable(forward):
            output = forward(batch)
        else:
            output = module._forward_inference(batch)
    logits = output[Columns.ACTION_DIST_INPUTS]
    return int(torch.argmax(logits, dim=-1).item())


def _get_module(algo):
    get_module = getattr(algo, "get_module", None)
    if callable(get_module):
        try:
            return get_module("default_policy")
        except Exception:
            return get_module()
    raise RuntimeError("Restored algorithm does not expose an RLModule.")


def _to_batched_tensors(obs: Dict[str, Any]) -> Dict[str, torch.Tensor]:
    batched: Dict[str, torch.Tensor] = {}
    for key, value in obs.items():
        tensor = torch.as_tensor(value)
        if key == "obs":
            tensor = tensor.float()
        elif key == "action_mask":
            tensor = tensor.float()
        else:
            tensor = tensor.long()
        if tensor.dim() >= 1:
            tensor = tensor.unsqueeze(0)
        batched[key] = tensor
    return batched


def _episode_stats(env) -> Dict[str, Any]:
    stats = env.pop_recent_episode_stats() if hasattr(env, "pop_recent_episode_stats") else []
    if stats:
        return dict(stats[-1])
    return {}


def _episode_outcome(
    episode_stats: Dict[str, Any],
    terminated: bool,
    truncated: bool,
) -> int:
    if "outcome" in episode_stats:
        return int(episode_stats["outcome"])
    if truncated or not terminated:
        return -1
    return -1
