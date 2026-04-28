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
from src.validation.teams import fixed_pair_battle_specs, load_team_manifest


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
    team_manifest: str | None = None,
    battle_format: str | None = None,
) -> Dict[str, Any]:
    """Restore a checkpoint and run a validation protocol."""
    if protocol.name not in {"smoke", "fixed_paired"}:
        raise NotImplementedError(
            f"Protocol '{protocol.name}' is planned but not implemented yet."
        )

    _seed_everything(seed)
    config = build_validation_config(preset)
    if battle_format:
        config.env.battle_format = battle_format
    checkpoint_path = resolve_resume_checkpoint(checkpoint, config.checkpoint_dir)
    if checkpoint_path is None:
        raise FileNotFoundError(
            f"Could not resolve checkpoint '{checkpoint}' in {config.checkpoint_dir}"
        )
    checkpoint_path = str(Path(checkpoint_path).expanduser().resolve())

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

        if protocol.name == "fixed_paired":
            if not team_manifest:
                raise ValueError("--team-manifest is required for fixed_paired.")
            manifest = load_team_manifest(team_manifest)
            execution_format = manifest.get("metadata", {}).get("execution_format")
            if isinstance(execution_format, str) and execution_format:
                config.env.battle_format = execution_format
            battle_specs = fixed_pair_battle_specs(manifest)
            results = _run_battle_specs(
                algo=algo,
                config=config,
                battle_specs=battle_specs,
                start_port=start_port,
                max_steps_per_battle=max_steps_per_battle,
            )
        else:
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
            "team_manifest": team_manifest,
            "battle_format": config.env.battle_format,
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
    player_team: str | None = None,
    opponent_team: str | None = None,
):
    env_creator = create_env_creator(
        battle_format=config.env.battle_format,
        server_host=config.env.showdown_host,
        server_port=start_port,
        reward_config=config.reward,
        opponent_difficulty=opponent_type,
        opponent_mix={opponent_type: 1.0},
        player_team=player_team,
        opponent_team=opponent_team,
    )
    return env_creator(
        {
            "server_port": start_port,
            "num_servers": 1,
            "start_port": start_port,
            "num_envs_per_worker": 1,
            "opponent_difficulty": opponent_type,
            "opponent_mix": {opponent_type: 1.0},
            "player_team": player_team,
            "opponent_team": opponent_team,
        }
    )


def _run_battle_specs(
    algo,
    config: TrainingConfig,
    battle_specs: List[Dict[str, Any]],
    start_port: int,
    max_steps_per_battle: int,
) -> List[BattleResult]:
    results: List[BattleResult] = []
    for episode_idx, spec in enumerate(battle_specs):
        print(
            f"Validation battle {episode_idx + 1}/{len(battle_specs)} | "
            f"{spec['pair_id']} | RL={spec['rl_team_id']} vs "
            f"{spec['opponent_type']}={spec['opponent_team_id']}",
            flush=True,
        )
        env = _build_validation_env(
            config=config,
            opponent_type=spec["opponent_type"],
            start_port=start_port,
            player_team=spec["rl_team"],
            opponent_team=spec["opponent_team"],
        )
        try:
            result = _run_one_episode(
                algo=algo,
                env=env,
                episode_idx=episode_idx,
                opponent_type=spec["opponent_type"],
                max_steps_per_battle=max_steps_per_battle,
                pair_id=spec["pair_id"],
                rl_team_id=spec["rl_team_id"],
                opponent_team_id=spec["opponent_team_id"],
            )
            results.append(result)
        finally:
            env.close()

    return results


def _run_episodes(
    algo,
    env,
    protocol: ValidationProtocol,
    max_steps_per_battle: int,
) -> List[BattleResult]:
    results: List[BattleResult] = []
    for episode_idx in range(protocol.episodes):
        results.append(
            _run_one_episode(
                algo=algo,
                env=env,
                episode_idx=episode_idx,
                opponent_type=protocol.opponent,
                max_steps_per_battle=max_steps_per_battle,
            )
        )

    return results


def _run_one_episode(
    algo,
    env,
    episode_idx: int,
    opponent_type: str,
    max_steps_per_battle: int,
    pair_id: str | None = None,
    rl_team_id: str | None = None,
    opponent_team_id: str | None = None,
) -> BattleResult:
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
    return BattleResult(
        episode=episode_idx,
        opponent_type=opponent_type,
        outcome=outcome,
        total_reward=total_reward,
        steps=steps,
        fallback_events=int(episode_stats.get("episode_fallback_events", 0)),
        attack_actions=int(episode_stats.get("episode_attack_actions", 0)),
        switch_actions=int(episode_stats.get("episode_switch_actions", 0)),
        pair_id=pair_id,
        rl_team_id=rl_team_id,
        opponent_team_id=opponent_team_id,
    )


def _compute_action(algo, obs: Dict[str, Any]) -> np.int64:
    compute_single_action = getattr(algo, "compute_single_action", None)
    if callable(compute_single_action):
        try:
            action = compute_single_action(obs, explore=False)
            if isinstance(action, tuple):
                action = action[0]
            return np.int64(action)
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
    return np.int64(torch.argmax(logits, dim=-1).item())


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
