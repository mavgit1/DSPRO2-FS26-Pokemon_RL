import gymnasium as gym
import torch
from typing import Optional
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.core.rl_module.rl_module import RLModuleSpec
from ray.tune.registry import register_env

from src.config.TM_optimal_config import CurriculumStageConfig, TrainingConfig
from src.action_space import COMPRESSED_ACTION_SPACE_N
from src.envs.battle_env import create_env_creator, get_observation_space

POKEMON_BATTLE_ENV_NAME = "pokemon_battle"

def register_environments(
    config: TrainingConfig,
    num_servers: int,
    start_port: int,
    initial_stage: Optional[CurriculumStageConfig],
) -> None:
    env_creator = create_env_creator(
        battle_format=config.env.battle_format,
        server_host=config.env.showdown_host,
        server_port=start_port,
        reward_config=(initial_stage.reward_config if initial_stage else config.reward),
        opponent_mix=(initial_stage.opponent_mix if initial_stage else None),
        model_config_dict=config.model.to_dict(),
        selfplay_weights_path=config.selfplay_weights_path,
    )
    register_env(POKEMON_BATTLE_ENV_NAME, env_creator)


def build_ppo_config(
    config: TrainingConfig, start_port: int, num_servers: int
) -> PPOConfig:
    from src.models.battle_transformer import PokemonRLModule

    return (
        PPOConfig()
        .environment(
            env=POKEMON_BATTLE_ENV_NAME,
            env_config={
                "num_servers": num_servers,
                "start_port": start_port,
                "num_envs_per_worker": config.env.num_envs_per_worker,
            },
        )
        .framework("torch")
        .api_stack(
            enable_rl_module_and_learner=True,
            enable_env_runner_and_connector_v2=True,
        )
        .training(
            lr=config.ppo.lr,
            gamma=config.ppo.gamma,
            lambda_=config.ppo.lambda_,
            clip_param=config.ppo.clip_param,
            entropy_coeff=config.ppo.entropy_coeff,
            vf_loss_coeff=config.ppo.vf_loss_coeff,
            vf_clip_param=config.ppo.vf_clip_param,
            grad_clip=config.ppo.grad_clip,
            train_batch_size=config.ppo.train_batch_size,
            minibatch_size=config.ppo.sgd_minibatch_size,
            num_epochs=config.ppo.num_sgd_iter,
        )
        .rl_module(
            rl_module_spec=RLModuleSpec(
                module_class=PokemonRLModule,
                observation_space=get_observation_space(),
                action_space=gym.spaces.Discrete(COMPRESSED_ACTION_SPACE_N),
                model_config={
                    **config.model.to_dict(),
                    "custom_model_config": config.model.to_dict(),
                },
            )
        )
        .env_runners(
            num_env_runners=config.env.num_workers,
            num_envs_per_env_runner=config.env.num_envs_per_worker,
        )
        .learners(
            num_learners=torch.cuda.device_count()
            if torch.cuda.is_available() and torch.cuda.device_count() > 1
            else 0,
            num_gpus_per_learner=1 if torch.cuda.is_available() else 0,
        )
        .debugging(log_level="WARNING")
    )
