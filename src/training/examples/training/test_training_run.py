import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent.parent.parent))
from poke_env.player import RandomPlayer
from src.training.examples.architecture.SimpleRLPlayer import SimpleRLPlayer
from poke_env.environment.single_agent_wrapper import SingleAgentWrapper
import uuid
from poke_env.ps_client.account_configuration import AccountConfiguration

opponent = RandomPlayer(
    battle_format="gen8randombattle",
    account_configuration=AccountConfiguration(f"Rand_{uuid.uuid4().hex[:6]}", None)
)
test_env = SimpleRLPlayer(
    battle_format="gen8randombattle",
    account_configuration1=AccountConfiguration(f"RL_{uuid.uuid4().hex[:6]}", None),
    strict=False
)
gym_env = SingleAgentWrapper(test_env, opponent=opponent)

print("Starting environment testing loop...")
obs, info = gym_env.reset()
for i in range(10):
    action = gym_env.action_space.sample()
    obs, reward, terminated, truncated, info = gym_env.step(action)
    print(f"Step {i} completed. Reward: {reward}")
    if terminated or truncated:
        print("Battle finished, resetting environment...")
        obs, info = gym_env.reset()

print("Environment loop ran successfully!")
gym_env.close()
