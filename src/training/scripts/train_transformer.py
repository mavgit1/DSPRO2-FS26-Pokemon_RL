import sys
import os
import uuid
from pathlib import Path
from dotenv import load_dotenv

# Set up project root on sys.path
PROJECT_ROOT = str(Path(__file__).resolve().parent.parent.parent.parent)
sys.path.append(PROJECT_ROOT)

# Load .env from project root
load_dotenv(os.path.join(PROJECT_ROOT, '.env'))

import ray
from ray.tune.registry import register_env
from ray.rllib.core.rl_module.rl_module import RLModuleSpec
from ray.rllib.algorithms.ppo import PPOConfig

from src.training.utilities.ray_embedding import RayEmbeddingEnv
from src.training.architecture.ray_transformer1 import RayTransformer1RLModule
from poke_env.player import RandomPlayer
from poke_env.environment.single_agent_wrapper import SingleAgentWrapper
from poke_env.ps_client.account_configuration import AccountConfiguration


def env_creator(env_config):
    """Create a SingleAgentWrapper environment for RLlib."""
    opponent = RandomPlayer(
        battle_format="gen8randombattle",
        account_configuration=AccountConfiguration(
            f"Opp_{uuid.uuid4().hex[:6]}", None
        ),
    )
    env = RayEmbeddingEnv(
        battle_format="gen8randombattle",
        account_configuration1=AccountConfiguration(
            f"RL_{uuid.uuid4().hex[:6]}", None
        ),
        strict=False,
    )
    return SingleAgentWrapper(env, opponent)


def main():
    ray.init()

    env_name = "PokemonRayEmbeddingEnv-v0"
    register_env(env_name, env_creator)

    config = (
        PPOConfig()
        .environment(env_name)
        .framework("torch")
        .rl_module(
            rl_module_spec=RLModuleSpec(
                module_class=RayTransformer1RLModule,
                model_config={},
            )
        )
        .training(
            lr=5e-5,
            train_batch_size=4000,
            minibatch_size=128,
            num_epochs=10,
        )
        # num_env_runners=0 -> run env locally in the driver process
        # This avoids Ray worker serialization / import issues
        # and satisfies the minimal parallelism rule
        .env_runners(num_env_runners=0, num_envs_per_env_runner=1)
        .learners(num_learners=0, num_gpus_per_learner=int(os.environ.get("RLLIB_NUM_GPUS", "0")))
    )

    print("Starting training run...")
    algo = config.build_algo()

    for i in range(5):
        result = algo.train()
        reward_mean = result.get("env_runners", {}).get("episode_reward_mean", "N/A")
        print(f"Iteration {i}: reward_mean = {reward_mean}")

    algo.stop()
    ray.shutdown()


if __name__ == "__main__":
    main()
