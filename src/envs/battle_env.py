import gymnasium as gym
import numpy as np
from typing import Dict, Any, Optional, List
import uuid
import random

from poke_env.battle.abstract_battle import AbstractBattle
from poke_env.environment.singles_env import SinglesEnv
from poke_env.ps_client.server_configuration import (
    ServerConfiguration,
    LocalhostServerConfiguration,
)
from poke_env.player import RandomPlayer, SimpleHeuristicsPlayer
from poke_env.environment.single_agent_wrapper import SingleAgentWrapper
from poke_env.ps_client.account_configuration import AccountConfiguration

from src.models.embedding import (
    embed_battle,
    NUM_TOKENS,
    TOKEN_DIM,
    MAX_ID_VAL,
)
from src.config.TM_optimal_config import RewardConfig


# =============================================================================
# OBSERVATION SPACE
# =============================================================================

def get_observation_space() -> gym.spaces.Dict:
    """Create the observation space for the environment."""
    return gym.spaces.Dict({
        "obs": gym.spaces.Box(
            low=-1.0,
            high=10.0,
            shape=(NUM_TOKENS, TOKEN_DIM),
            dtype=np.float32,
        ),
        "species": gym.spaces.Box(
            low=0,
            high=MAX_ID_VAL,
            shape=(NUM_TOKENS,),
            dtype=np.int32,
        ),
        "items": gym.spaces.Box(
            low=0,
            high=MAX_ID_VAL,
            shape=(NUM_TOKENS,),
            dtype=np.int32,
        ),
        "abilities": gym.spaces.Box(
            low=0,
            high=MAX_ID_VAL,
            shape=(NUM_TOKENS,),
            dtype=np.int32,
        ),
        "action_mask": gym.spaces.Box(
            low=0,
            high=1,
            shape=(22,),
            dtype=np.float32,
        ),
    })


# =============================================================================
# BASE ENVIRONMENT 
# =============================================================================

class PokemonBattleEnv(SinglesEnv):
    """
    Gymnasium environment for Pokemon battles with transformer-friendly embeddings.
    
    Extends SinglesEnv (PettingZoo ParallelEnv) and sets observation_spaces
    as a dict keyed by agent usernames.
    
    Features:
        - Token-based observation space
        - Categorical embeddings for species, items, abilities
        - Action masking for valid actions
        - Configurable reward function
    """
    
    def __init__(
        self,
        reward_config: Optional[RewardConfig] = None,
        **kwargs
    ):
        """
        Initialize the environment.
        
        Args:
            reward_config: Reward configuration
            **kwargs: Passed to SinglesEnv (battle_format, account_configuration1,
                      server_configuration, strict, etc.)
        """
        self.reward_config = reward_config or RewardConfig()
        self._recent_outcomes: List[int] = []
        self._recent_episode_stats: List[Dict[str, float]] = []
        self._battle_step_stats: Dict[str, Dict[str, float]] = {}
        self._completed_battle_steps: Dict[str, int] = {}
        self._env_step_counter = 0
        self._stale_battle_step_ttl = 2048
        self._completed_battle_ttl = 4096
        self._cleanup_interval_steps = 256
        self._fallback_events_current_episode = 0
        
        super().__init__(**kwargs)
        
        # PettingZoo-style observation_spaces dict keyed by agent
        obs_space = get_observation_space()
        self.observation_spaces = {
            agent: obs_space
            for agent in self.possible_agents
        }
    
    def embed_battle(self, battle: AbstractBattle) -> Dict[str, np.ndarray]:
        """
        Convert battle state to embedding.
        
        Args:
            battle: Current battle state
        
        Returns:
            Dict with obs, species, items, abilities, action_mask
        """
        return embed_battle(battle)
    
    def calc_reward(self, battle: AbstractBattle) -> float:
        """Calculate reward based on battle state."""
        battle_tag = getattr(battle, "battle_tag", None)
        battle_key = str(battle_tag) if battle_tag else f"battle_{id(battle)}"
        self._env_step_counter += 1
        if self._env_step_counter % self._cleanup_interval_steps == 0:
            self._prune_stale_tracking()

        if battle_key not in self._completed_battle_steps:
            step_stats = self._battle_step_stats.setdefault(
                battle_key,
                {
                    "action_mask_valid_sum": 0.0,
                    "action_mask_count": 0.0,
                    "last_seen_step": 0.0,
                },
            )

            # Track valid-action density during the episode.
            action_mask = embed_battle(battle)["action_mask"]
            step_stats["action_mask_valid_sum"] += float(np.sum(action_mask))
            step_stats["action_mask_count"] += 1.0
            step_stats["last_seen_step"] = float(self._env_step_counter)

            if battle.won:
                self._recent_outcomes.append(1)
                self._completed_battle_steps[battle_key] = self._env_step_counter
                self._recent_episode_stats.append(
                    self._build_terminal_episode_stats(battle, battle_key, outcome=1)
                )
            elif battle.lost:
                self._recent_outcomes.append(0)
                self._completed_battle_steps[battle_key] = self._env_step_counter
                self._recent_episode_stats.append(
                    self._build_terminal_episode_stats(battle, battle_key, outcome=0)
                )
        elif battle.won or battle.lost:
            # Refresh terminal marker while reward callbacks are still firing.
            self._completed_battle_steps[battle_key] = self._env_step_counter

        return self.reward_computing_helper(
            battle,
            fainted_value=self.reward_config.fainted_value,
            hp_value=self.reward_config.hp_value_weight,
            victory_value=self.reward_config.victory_reward,
        )

    def set_reward_config(self, reward_config: RewardConfig) -> None:
        """Update reward configuration at runtime."""
        self.reward_config = reward_config

    def pop_recent_outcomes(self) -> List[int]:
        """Return and clear terminal battle outcomes (1 win, 0 loss)."""
        outcomes = self._recent_outcomes[:]
        self._recent_outcomes.clear()
        return outcomes

    def pop_recent_episode_stats(self) -> List[Dict[str, float]]:
        """Return and clear per-episode summary stats."""
        stats = self._recent_episode_stats[:]
        self._recent_episode_stats.clear()
        return stats

    def consume_fallback_events(self) -> int:
        """Return and reset conversion fallback events for current episode."""
        events = int(self._fallback_events_current_episode)
        self._fallback_events_current_episode = 0
        return events

    def _prune_stale_tracking(self) -> None:
        """Drop stale battle bookkeeping for interrupted/disconnected episodes."""
        active_cutoff = self._env_step_counter - self._stale_battle_step_ttl
        stale_active = [
            key
            for key, stats in self._battle_step_stats.items()
            if int(stats.get("last_seen_step", 0.0)) < active_cutoff
        ]
        for key in stale_active:
            self._battle_step_stats.pop(key, None)

        completed_cutoff = self._env_step_counter - self._completed_battle_ttl
        stale_completed = [
            key
            for key, seen_step in self._completed_battle_steps.items()
            if int(seen_step) < completed_cutoff
        ]
        for key in stale_completed:
            self._completed_battle_steps.pop(key, None)

    def reset_tracking_state(self) -> None:
        """Clear episode/battle-local tracking to avoid cross-episode retention."""
        self._battle_step_stats.clear()
        self._completed_battle_steps.clear()

    def get_memory_counters(self) -> Dict[str, float]:
        """Small diagnostics payload for leak monitoring."""
        return {
            "battle_step_stats_len": float(len(self._battle_step_stats)),
            "completed_battle_markers_len": float(len(self._completed_battle_steps)),
            "recent_outcomes_len": float(len(self._recent_outcomes)),
            "recent_episode_stats_len": float(len(self._recent_episode_stats)),
        }

    def _build_terminal_episode_stats(
        self, battle: AbstractBattle, battle_key: str, outcome: int
    ) -> Dict[str, float]:
        our_hp = _get_team_hp_fraction(battle.team)
        opp_hp = _get_team_hp_fraction(battle.opponent_team)
        our_fainted = sum(1 for m in battle.team.values() if m.fainted)
        opp_fainted = sum(1 for m in battle.opponent_team.values() if m.fainted)
        hp_diff = our_hp - opp_hp

        reward_victory = (
            self.reward_config.victory_reward if outcome == 1 else self.reward_config.defeat_penalty
        )
        reward_hp_diff = hp_diff * self.reward_config.hp_value_weight
        reward_faint = (
            opp_fainted * self.reward_config.fainted_value
            - our_fainted * self.reward_config.fainted_penalty
        )
        battle_turns = float(max(0, int(getattr(battle, "turn", 0))))
        reward_step = battle_turns * self.reward_config.step_penalty

        step_stats = self._battle_step_stats.pop(
            battle_key,
            {
                "action_mask_valid_sum": 0.0,
                "action_mask_count": 0.0,
                "last_seen_step": 0.0,
            },
        )
        mask_count = max(step_stats["action_mask_count"], 1.0)
        mask_valid_mean = step_stats["action_mask_valid_sum"] / mask_count

        return {
            "outcome": float(outcome),
            "terminal_our_hp_remaining": float(our_hp),
            "terminal_opp_hp_remaining": float(opp_hp),
            "terminal_faint_diff": float(opp_fainted - our_fainted),
            "battle_turns": battle_turns,
            "reward_victory_component": float(reward_victory),
            "reward_hp_diff_component": float(reward_hp_diff),
            "reward_faint_component": float(reward_faint),
            "reward_step_penalty_component": float(reward_step),
            "action_mask_valid_count_mean": float(mask_valid_mean),
        }

    def order_to_action(self, order, battle, fake: bool = False, strict: bool = True):
        """
        Convert a BattleOrder to action index with bounded fallbacks.

        poke-env's default strict=False path can recurse indefinitely if random
        fallback orders keep failing conversion. We cap retries and then choose a
        guaranteed legal action id by probing action_to_order.
        """
        try:
            return SinglesEnv.order_to_action(order, battle, fake=fake, strict=True)
        except ValueError:
            if strict:
                raise

        # Retry with random legal-looking orders a fixed number of times.
        max_retries = 5
        for _ in range(max_retries):
            random_order = RandomPlayer.choose_random_singles_move(battle)
            try:
                self._fallback_events_current_episode += 1
                return SinglesEnv.order_to_action(
                    random_order, battle, fake=fake, strict=True
                )
            except ValueError:
                continue

        # Hard fallback: pick the first action that converts legally.
        # 26 covers up to gen9 singles action size; gen8 uses 22.
        for action in range(26):
            try:
                self._fallback_events_current_episode += 1
                SinglesEnv.action_to_order(
                    np.int64(action), battle, fake=fake, strict=True
                )
                return np.int64(action)
            except ValueError:
                continue

        # If no legal action could be verified, return default action.
        self._fallback_events_current_episode += 1
        return np.int64(-2)


class CurriculumSingleAgentWrapper(SingleAgentWrapper):
    """Single-agent wrapper that supports opponent-mix curriculum updates."""

    def __init__(
        self,
        env: PokemonBattleEnv,
        opponent,
        battle_format: str,
        server_configuration: ServerConfiguration,
        opponent_mix: Optional[Dict[str, float]] = None,
    ):
        super().__init__(env, opponent)
        self._battle_format = battle_format
        self._server_configuration = server_configuration
        self._opponent_mix = self._normalize_opponent_mix(opponent_mix)
        self._opponent_pool: Dict[str, Any] = {}
        self._episode_total_actions = 0
        self._episode_switch_actions = 0
        self._episode_attack_actions = 0
        self._recent_action_stats: List[Dict[str, float]] = []

        initial_key = self._opponent_key_from_instance(opponent)
        self._opponent_pool[initial_key] = opponent

    @staticmethod
    def _normalize_opponent_mix(opponent_mix: Optional[Dict[str, float]]) -> Dict[str, float]:
        default_mix = {"random": 1.0}
        if not opponent_mix:
            return default_mix

        valid = {}
        for key, val in opponent_mix.items():
            key_lower = str(key).strip().lower()
            if key_lower in {"random", "heuristic", "heuristics"} and float(val) > 0:
                canonical = "heuristic" if key_lower == "heuristics" else key_lower
                valid[canonical] = valid.get(canonical, 0.0) + float(val)

        total = sum(valid.values())
        if total <= 0:
            return default_mix
        return {k: v / total for k, v in valid.items()}

    def _choose_opponent_class(self):
        keys = list(self._opponent_mix.keys())
        weights = [self._opponent_mix[k] for k in keys]
        selected = random.choices(keys, weights=weights, k=1)[0]
        return "heuristic" if selected == "heuristic" else "random"

    def _build_opponent(self, opponent_key: str):
        opponent_class = (
            SimpleHeuristicsPlayer if opponent_key == "heuristic" else RandomPlayer
        )
        opponent_id = f"Opp_{uuid.uuid4().hex[:6]}"
        opponent_config = AccountConfiguration(opponent_id, None)
        return opponent_class(
            battle_format=self._battle_format,
            account_configuration=opponent_config,
            server_configuration=self._server_configuration,
        )

    @staticmethod
    def _opponent_key_from_instance(opponent: Any) -> str:
        if isinstance(opponent, SimpleHeuristicsPlayer):
            return "heuristic"
        return "random"

    @staticmethod
    def _close_opponent(opponent) -> None:
        close_fn = getattr(opponent, "close", None)
        if callable(close_fn):
            try:
                close_fn()
            except Exception:
                # Opponent teardown should be best-effort only.
                pass

    def reset(self, *args, **kwargs):
        # Sample an opponent per episode according to configured mix.
        opponent_key = self._choose_opponent_class()
        if opponent_key not in self._opponent_pool:
            self._opponent_pool[opponent_key] = self._build_opponent(opponent_key)
        self.opponent = self._opponent_pool[opponent_key]
        self._episode_total_actions = 0
        self._episode_switch_actions = 0
        self._episode_attack_actions = 0
        if hasattr(self.env, "reset_tracking_state"):
            self.env.reset_tracking_state()
        if hasattr(self.env, "consume_fallback_events"):
            self.env.consume_fallback_events()
        return super().reset(*args, **kwargs)

    def step(self, action):
        if action is not None:
            action_int = int(action)
            self._episode_total_actions += 1
            # In poke-env singles indexing, switches are in upper action ids.
            if action_int >= 16:
                self._episode_switch_actions += 1
            else:
                self._episode_attack_actions += 1

        result = super().step(action)
        terminated = False
        truncated = False
        if isinstance(result, tuple):
            if len(result) == 5:
                terminated = bool(result[2])
                truncated = bool(result[3])
            elif len(result) == 4:
                terminated = bool(result[2])

        if terminated or truncated:
            fallback_events = 0
            if hasattr(self.env, "consume_fallback_events"):
                fallback_events = int(self.env.consume_fallback_events())
            self._recent_action_stats.append(
                {
                    "episode_total_actions": float(self._episode_total_actions),
                    "episode_switch_actions": float(self._episode_switch_actions),
                    "episode_attack_actions": float(self._episode_attack_actions),
                    "episode_fallback_events": float(fallback_events),
                }
            )
        return result

    def set_opponent_mix(self, opponent_mix: Dict[str, float]) -> None:
        self._opponent_mix = self._normalize_opponent_mix(opponent_mix)

    def set_reward_config(self, reward_config: RewardConfig) -> None:
        if hasattr(self.env, "set_reward_config"):
            self.env.set_reward_config(reward_config)

    def apply_curriculum_stage(self, stage_payload: Dict[str, Any]) -> None:
        if "opponent_mix" in stage_payload:
            self.set_opponent_mix(stage_payload["opponent_mix"])
        if "reward_config" in stage_payload:
            self.set_reward_config(RewardConfig(**stage_payload["reward_config"]))

    def pop_recent_outcomes(self) -> List[int]:
        if hasattr(self.env, "pop_recent_outcomes"):
            return self.env.pop_recent_outcomes()
        return []

    def pop_recent_episode_stats(self) -> List[Dict[str, float]]:
        env_stats = []
        if hasattr(self.env, "pop_recent_episode_stats"):
            env_stats = self.env.pop_recent_episode_stats()

        action_stats = self._recent_action_stats[:]
        self._recent_action_stats.clear()

        count = min(len(env_stats), len(action_stats))
        merged = []
        for idx in range(count):
            item = dict(env_stats[idx])
            item.update(action_stats[idx])
            merged.append(item)
        return merged

    def get_memory_counters(self) -> Dict[str, float]:
        out = {
            "wrapper_recent_action_stats_len": float(len(self._recent_action_stats)),
            "wrapper_opponent_pool_len": float(len(self._opponent_pool)),
        }
        if hasattr(self.env, "get_memory_counters"):
            env_counters = self.env.get_memory_counters()
            for key, value in env_counters.items():
                out[f"env_{key}"] = float(value)
        return out

    def close(self):
        for opponent in self._opponent_pool.values():
            self._close_opponent(opponent)
        self._opponent_pool.clear()
        if hasattr(self.env, "reset_tracking_state"):
            self.env.reset_tracking_state()
        return super().close()

# =============================================================================
# REWARD FUNCTION todo: create more for different curriculum stages
# =============================================================================

def compute_reward(battle: AbstractBattle, config: RewardConfig) -> float:
    """
    Compute reward based on battle state and configuration.
    
    Args:
        battle: Current battle state
        config: Reward configuration
    
    Returns:
        Float reward value
    """
    reward = 0.0
    
    # Victory/Loss (terminal)
    if battle.won:
        return config.victory_reward
    if battle.lost:
        return config.defeat_penalty
    
    # HP-based reward
    our_hp = _get_team_hp_fraction(battle.team)
    opp_hp = _get_team_hp_fraction(battle.opponent_team)
    
    hp_diff = our_hp - opp_hp
    reward += hp_diff * config.hp_value_weight
    
    # Fainting rewards
    our_fainted = sum(1 for m in battle.team.values() if m.fainted)
    opp_fainted = sum(1 for m in battle.opponent_team.values() if m.fainted)
    
    reward += opp_fainted * config.fainted_value
    reward -= our_fainted * config.fainted_penalty
    
    # Step penalty (encourage efficiency)
    reward += config.step_penalty
    
    return reward


def _get_team_hp_fraction(team: Dict) -> float:
    """Get total HP fraction for a team."""
    total = 0.0
    for mon in team.values():
        if not mon.fainted:
            total += mon.current_hp_fraction
    return total


# =============================================================================
# ENVIRONMENT CREATOR FOR RAY
# =============================================================================

def create_env_creator(
    battle_format: str = "gen8randombattle",
    server_host: str = "localhost",
    server_port: int = 8000,
    reward_config: Optional[RewardConfig] = None,
    opponent_difficulty: str = "heuristic",
    opponent_mix: Optional[Dict[str, float]] = None,
):
    """
    Create an environment creator function for Ray RLlib.
    
    Args:
        battle_format: Battle format string
        server_host: Showdown server host
        server_port: Showdown server port
        reward_config: Reward configuration
        opponent_difficulty: "heuristic"/"heuristics" or "random"
        opponent_mix: Optional per-episode sampling mix, e.g. {"random": 0.7, "heuristic": 0.3}
    
    Returns:
        Callable that creates environments
    """
    def env_creator(env_config: Optional[Dict] = None):
        env_config = env_config or {}
        
        # Resolve settings
        fmt = env_config.get("battle_format", battle_format)
        host = env_config.get("server_host", server_host)
        port = env_config.get("server_port", server_port)
        rc = env_config.get("reward_config", reward_config or RewardConfig())
        difficulty = env_config.get("opponent_difficulty", opponent_difficulty)
        mix = env_config.get("opponent_mix", opponent_mix)
        
        # Build proper websocket ServerConfiguration
        server_config = ServerConfiguration(
            f"ws://{host}:{port}/showdown/websocket",
            "https://play.pokemonshowdown.com/action.php?",
        )
        
        # Create a starting opponent. Wrapper will resample per episode
        # when opponent mixes are configured.
        opponent_id = f"Opp_{uuid.uuid4().hex[:6]}"
        opponent_config = AccountConfiguration(opponent_id, None)
        if difficulty in {"heuristic", "heuristics"}:
            opponent_class = SimpleHeuristicsPlayer
        else:
            opponent_class = RandomPlayer
        opponent = opponent_class(
            battle_format=fmt,
            account_configuration=opponent_config,
            server_configuration=server_config,
        )
        
        # Create the PettingZoo env
        player_id = f"RL_{uuid.uuid4().hex[:8]}"
        env = PokemonBattleEnv(
            reward_config=rc,
            battle_format=fmt,
            account_configuration1=AccountConfiguration(player_id, None),
            server_configuration=server_config,
            strict=False,
        )
        
        # Wrap into single-agent gym env
        return CurriculumSingleAgentWrapper(
            env=env,
            opponent=opponent,
            battle_format=fmt,
            server_configuration=server_config,
            opponent_mix=mix,
        )
    
    return env_creator