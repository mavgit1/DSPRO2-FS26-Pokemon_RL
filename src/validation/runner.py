from __future__ import annotations

import random
import socket
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
from src.validation.metrics import (
    BattleResult,
    aggregate_validation_metrics,
    build_validation_diagnostics,
)
from src.validation.protocols import ValidationProtocol
from src.validation.teams import (
    fixed_pair_battle_specs,
    load_team_manifest,
    mirror_battle_specs,
)


def build_validation_config(preset: str) -> TrainingConfig:
    """Build a lightweight config for checkpoint validation."""
    config = get_config(preset)
    config.env.num_workers = 0
    config.env.num_envs_per_worker = 1
    config.ppo.train_batch_size = min(config.ppo.train_batch_size, 512)
    config.ppo.sgd_minibatch_size = min(config.ppo.sgd_minibatch_size, 128)
    return config


_VALIDATION_RESTORE_COMPONENTS = (
    "env_runner/rl_module",
    "learner_group/learner/rl_module",
)


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
    use_lstm: bool = False,
    player_team_path: str | None = None,
) -> Dict[str, Any]:
    """Restore a checkpoint and run a validation protocol."""
    if protocol.name not in {"smoke", "fixed_paired", "mirror"}:
        raise NotImplementedError(
            f"Protocol '{protocol.name}' is planned but not implemented yet."
        )

    _seed_everything(seed)
    config = build_validation_config(preset)
    config.model.use_lstm = use_lstm

    # Load fixed player team if provided.
    player_team: str | None = None
    if player_team_path:
        player_team = Path(player_team_path).expanduser().resolve().read_text(encoding="utf-8").strip()

    if battle_format:
        config.env.battle_format = battle_format
    checkpoint_path = resolve_resume_checkpoint(checkpoint, config.checkpoint_dir)
    if checkpoint_path is None:
        raise FileNotFoundError(
            f"Could not resolve checkpoint '{checkpoint}' in {config.checkpoint_dir}"
        )
    checkpoint_path_obj = Path(checkpoint_path).expanduser().resolve()
    if not checkpoint_path_obj.exists():
        raise FileNotFoundError(
            f"Checkpoint path does not exist: {checkpoint_path_obj}"
        )
    checkpoint_path = str(checkpoint_path_obj)
    _ensure_showdown_server(config.env.showdown_host, start_port)

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
        try:
            _restore_checkpoint_for_validation(algo, checkpoint_path)
        except RuntimeError:
            if use_lstm:
                raise
            algo.stop()
            config.model.use_lstm = True
            algo = build_ppo_config(
                config=config,
                start_port=start_port,
                num_servers=num_servers,
            ).build_algo()
            _restore_checkpoint_for_validation(algo, checkpoint_path)

        if protocol.name in {"fixed_paired", "mirror"}:
            if protocol.name == "mirror" and player_team:
                # Fixed-team mirror: same team for both sides, against
                # random and heuristic opponents.
                from src.validation.teams import fixed_team_mirror_specs
                execution_format = config.env.battle_format.replace(
                    "randombattle", "customgame"
                )
                config.env.battle_format = execution_format
                battle_specs = fixed_team_mirror_specs(player_team)
            else:
                if not team_manifest:
                    raise ValueError(f"--team-manifest is required for {protocol.name}.")
                manifest = load_team_manifest(team_manifest)
                execution_format = manifest.get("metadata", {}).get("execution_format")
                if isinstance(execution_format, str) and execution_format:
                    config.env.battle_format = execution_format
                if protocol.name == "fixed_paired":
                    battle_specs = fixed_pair_battle_specs(manifest)
                    # Override RL team with fixed player team if set.
                    if player_team:
                        for spec in battle_specs:
                            spec["rl_team"] = player_team
                            spec["rl_team_id"] = "player_team"
                else:
                    battle_specs = mirror_battle_specs(manifest)
            results = _run_battle_specs(
                algo=algo,
                config=config,
                battle_specs=battle_specs,
                start_port=start_port,
                max_steps_per_battle=max_steps_per_battle,
            )
        else:
            # Auto-switch to customgame when a player team is set.
            if player_team:
                config.env.battle_format = config.env.battle_format.replace(
                    "randombattle", "customgame"
                )
            env = _build_validation_env(
                config=config,
                opponent_type=protocol.opponent,
                start_port=start_port,
                player_team=player_team,
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
        "diagnostics": build_validation_diagnostics(results),
        "episodes": [result.to_dict() for result in results],
    }


def _restore_checkpoint_for_validation(algo, checkpoint_path: str) -> None:
    """Restore only inference weights for checkpoint validation.

    Full Algorithm restore includes learner optimizer state. Validation never
    uses the optimizer, and restoring it is brittle when the validation build
    intentionally changes learner-only settings or when checkpoints were
    created before small model/config edits. Restoring the RLModule component
    is enough because validation reads actions directly from the module.
    """
    restore_errors: list[str] = []
    for component in _VALIDATION_RESTORE_COMPONENTS:
        try:
            algo.restore_from_path(checkpoint_path, component=component)
            return
        except Exception as exc:
            restore_errors.append(f"{component}: {exc}")

    try:
        algo.restore(checkpoint_path)
    except Exception as exc:
        restore_errors.append(f"full algorithm restore: {exc}")
        joined = "\n".join(f"- {error}" for error in restore_errors)
        raise RuntimeError(
            "Could not restore checkpoint for validation. Ensure the validation "
            "preset and --use-lstm flag match the checkpoint's training config.\n"
            f"Restore attempts:\n{joined}"
        ) from exc


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _ensure_showdown_server(host: str, port: int) -> None:
    """Fail fast if the expected Pokemon Showdown server is unavailable."""
    try:
        with socket.create_connection((host, port), timeout=2.0):
            return
    except OSError as exc:
        raise ConnectionError(
            f"Pokemon Showdown server is not reachable at {host}:{port}. "
            "Start the server before running validation."
        ) from exc


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
    recurrent_state = None

    while not terminated and not truncated and steps < max_steps_per_battle:
        action, recurrent_state = _compute_action(algo, obs, recurrent_state)
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


def _compute_action(
    algo,
    obs: Dict[str, Any],
    recurrent_state: Dict[str, torch.Tensor] | None = None,
) -> tuple[np.int64, Dict[str, torch.Tensor] | None]:
    module = _get_module(algo)
    if _module_is_stateful(module):
        return _compute_recurrent_action(module, obs, recurrent_state)

    compute_single_action = getattr(algo, "compute_single_action", None)
    if callable(compute_single_action):
        try:
            action = compute_single_action(obs, explore=False)
            if isinstance(action, tuple):
                action = action[0]
            return np.int64(action), None
        except Exception:
            pass

    batch = {Columns.OBS: _to_batched_tensors(obs)}
    with torch.no_grad():
        forward = getattr(module, "forward_inference", None)
        if callable(forward):
            output = forward(batch)
        else:
            output = module._forward_inference(batch)
    logits = output[Columns.ACTION_DIST_INPUTS]
    return np.int64(torch.argmax(logits, dim=-1).item()), None


def _compute_recurrent_action(
    module,
    obs: Dict[str, Any],
    recurrent_state: Dict[str, torch.Tensor] | None,
) -> tuple[np.int64, Dict[str, torch.Tensor]]:
    """Run direct validation inference for a stateful RLModule.

    Validation does not use RLlib's env-runner connector stack, so we must do
    the two recurrent connector jobs manually: add a single-step time rank to
    observations and feed STATE_OUT back as the next STATE_IN.
    """
    if recurrent_state is None:
        recurrent_state = _initial_state_as_batched_tensors(module)

    batch = {
        Columns.OBS: _to_batched_tensors(obs, add_time_dim=True),
        Columns.STATE_IN: recurrent_state,
    }
    with torch.no_grad():
        forward = getattr(module, "forward_inference", None)
        if callable(forward):
            output = forward(batch)
        else:
            output = module._forward_inference(batch)

    logits = output[Columns.ACTION_DIST_INPUTS]
    action = np.int64(torch.argmax(logits, dim=-1).item())
    next_state = {
        key: value.detach() for key, value in output[Columns.STATE_OUT].items()
    }
    return action, next_state


def _get_module(algo):
    get_module = getattr(algo, "get_module", None)
    if callable(get_module):
        try:
            return get_module("default_policy")
        except Exception:
            return get_module()
    raise RuntimeError("Restored algorithm does not expose an RLModule.")


def _module_is_stateful(module) -> bool:
    is_stateful = getattr(module, "is_stateful", None)
    return bool(is_stateful()) if callable(is_stateful) else False


def _initial_state_as_batched_tensors(module) -> Dict[str, torch.Tensor]:
    initial_state = module.get_initial_state()
    device = _module_device(module)
    return {
        key: torch.as_tensor(value, dtype=torch.float32, device=device).unsqueeze(0)
        for key, value in initial_state.items()
    }


def _module_device(module) -> torch.device:
    try:
        return next(module.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def _to_batched_tensors(
    obs: Dict[str, Any],
    *,
    add_time_dim: bool = False,
) -> Dict[str, torch.Tensor]:
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
        if add_time_dim:
            tensor = tensor.unsqueeze(1)
        batched[key] = tensor
    return batched


def _episode_stats(env) -> Dict[str, Any]:
    stats = (
        env.pop_recent_episode_stats()
        if hasattr(env, "pop_recent_episode_stats")
        else []
    )
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
