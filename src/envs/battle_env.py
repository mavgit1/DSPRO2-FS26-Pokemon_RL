import gymnasium as gym
import numpy as np
from typing import Dict, Any, Optional, List
import uuid
import random
import re

from poke_env.battle.abstract_battle import AbstractBattle
from poke_env.environment.singles_env import SinglesEnv
from poke_env.ps_client.server_configuration import ServerConfiguration
from poke_env.player import RandomPlayer, SimpleHeuristicsPlayer
from poke_env.environment.single_agent_wrapper import SingleAgentWrapper
from poke_env.ps_client.account_configuration import AccountConfiguration

from src.action_space import NATIVE_ACTION_SPACE_N, is_native_switch_action


def _native_move_slot(action: int) -> Optional[int]:
    """Move index 0-3 for native attack actions (6-21), else None."""
    action = int(action)
    if 6 <= action <= 21:
        return (action - 6) % 4
    return None


def _move_type_effectiveness(move: Any, target_types: tuple) -> float:
    move_type = getattr(move, "type", None)
    if move_type is None:
        return 1.0
    try:
        return float(move_type.damage_multiplier(*target_types))
    except Exception:
        return 1.0


def _is_damaging_move(move: Any) -> bool:
    category = getattr(move, "category", None)
    if category is None:
        return True
    return getattr(category, "name", "") != "STATUS"


def _best_damaging_effectiveness(moves: Any, target_types: tuple) -> float:
    best = 0.0
    found = False
    for move in moves.values() if hasattr(moves, "values") else moves:
        if not _is_damaging_move(move):
            continue
        eff = _move_type_effectiveness(move, target_types)
        best = max(best, eff)
        found = True
    return best if found else 0.0
from src.models.embedding import (
    embed_battle,
    NUM_TOKENS,
    TOKEN_DIM,
    SPECIES_VOCAB_SIZE,
    ITEM_VOCAB_SIZE,
    ABILITY_VOCAB_SIZE,
)
from src.config.TM_optimal_config import RewardConfig


# =============================================================================
# OBSERVATION SPACE
# =============================================================================


def get_observation_space() -> gym.spaces.Dict:
    """Create the observation space for the environment."""
    return gym.spaces.Dict(
        {
            "obs": gym.spaces.Box(
                low=-1.0,
                high=10.0,
                shape=(NUM_TOKENS, TOKEN_DIM),
                dtype=np.float32,
            ),
            "species": gym.spaces.Box(
                low=0,
                high=SPECIES_VOCAB_SIZE - 1,
                shape=(NUM_TOKENS,),
                dtype=np.int32,
            ),
            "items": gym.spaces.Box(
                low=0,
                high=ITEM_VOCAB_SIZE - 1,
                shape=(NUM_TOKENS,),
                dtype=np.int32,
            ),
            "abilities": gym.spaces.Box(
                low=0,
                high=ABILITY_VOCAB_SIZE - 1,
                shape=(NUM_TOKENS,),
                dtype=np.int32,
            ),
            "action_mask": gym.spaces.Box(
                low=0,
                high=1,
                shape=(NATIVE_ACTION_SPACE_N,),
                dtype=np.float32,
            ),
        }
    )


def get_rllib_observation_space() -> gym.spaces.Dict:
    """Observation space as returned by poke-env 0.15+ (embed dict nested under ``observation``)."""
    return gym.spaces.Dict(
        {
            "observation": get_observation_space(),
            "action_mask": gym.spaces.Box(
                low=0,
                high=1,
                shape=(NATIVE_ACTION_SPACE_N,),
                dtype=np.float32,
            ),
        }
    )


def flatten_agent_observation(obs: Any) -> Optional[Dict[str, np.ndarray]]:
    """Unwrap poke-env's ``{observation: {...}, action_mask}`` layout for the model."""
    if not isinstance(obs, dict):
        return None
    if "obs" in obs:
        return obs
    inner = obs.get("observation")
    if not isinstance(inner, dict) or "obs" not in inner:
        return None
    flat = dict(inner)
    if "action_mask" in obs:
        flat["action_mask"] = obs["action_mask"]
    return flat


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

    def __init__(self, reward_config: Optional[RewardConfig] = None, **kwargs):
        """
        Initialize the environment.

        Args:
            reward_config: Reward configuration
            **kwargs: Passed to SinglesEnv (battle_format, account_configuration1,
                      server_configuration, strict, etc.)
        """
        self.reward_config = reward_config or RewardConfig()
        self._last_native_action: int = -1
        self._recent_outcomes: List[int] = []
        self._recent_episode_stats: List[Dict[str, float]] = []
        self._battle_step_stats: Dict[str, Dict[str, float]] = {}
        self._completed_battle_steps: Dict[str, int] = {}
        self._env_step_counter = 0
        self._stale_battle_step_ttl = 2048
        self._completed_battle_ttl = 4096
        self._cleanup_interval_steps = 256
        self._fallback_events_current_episode = 0
        self._opponent_context: Optional[str] = None
        self._training_stage_context: int = 0

        super().__init__(**kwargs)

        # PettingZoo-style observation_spaces dict keyed by agent
        obs_space = get_observation_space()
        self.observation_spaces = {agent: obs_space for agent in self.possible_agents}

    def embed_battle(self, battle: AbstractBattle) -> Dict[str, np.ndarray]:
        """
        Convert battle state to embedding.

        Args:
            battle: Current battle state

        Returns:
            Dict with obs, species, items, abilities, action_mask
        """
        # Not recursive, just calls the embed_battle function from the embedding.py file.
        return embed_battle(
            battle,
            opponent_type=self._opponent_context,
            training_stage_index=self._training_stage_context,
        )

    def set_opponent_context(self, opponent_type: Optional[str]) -> None:
        """Attach selected opponent metadata to future observations."""
        self._opponent_context = opponent_type

    def set_training_stage_context(self, stage_index: int) -> None:
        """Attach curriculum/training stage index to future observations."""
        self._training_stage_context = max(0, int(stage_index))

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
            action_mask = self.embed_battle(battle)["action_mask"]
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

        reward = float(self._compute_configured_delta_reward(battle))
        
        if getattr(self, "_step_fallback_penalty", False):
            reward -= 0.5
            self._step_fallback_penalty = False
            
        return reward

    def _compute_configured_delta_reward(self, battle: AbstractBattle) -> float:
        """Poke-env style delta reward with matchup shaping."""
        if battle not in self._reward_buffer:
            self._reward_buffer[battle] = 0.0

        state_value = 0.0
        hp_value = self.reward_config.hp_value_weight
        fainted_value = self.reward_config.fainted_value
        fainted_penalty = self.reward_config.fainted_penalty
        number_of_pokemons = 6

        # Calculate OUR Persistent State
        our_hp = 0.0
        for mon in battle.team.values():
            our_hp += mon.current_hp_fraction
            if mon.fainted:
                state_value += fainted_penalty
        
        # Unrevealed own Pokemon
        our_hp += (number_of_pokemons - len(battle.team))

        # Calculate OPPONENT Persistent State
        opp_hp = 0.0
        for mon in battle.opponent_team.values():
            opp_hp += mon.current_hp_fraction
            if mon.fainted:
                state_value += fainted_value
                
        # Unrevealed opponent Pokemon
        opp_hp += (number_of_pokemons - len(battle.opponent_team))

        # Apply HP weights
        state_value += our_hp * hp_value
        state_value -= opp_hp * hp_value

        # Calculate Delta
        reward_delta = state_value - self._reward_buffer[battle]
        self._reward_buffer[battle] = state_value

        # Ephemeral Step Rewards
        step_reward = 0.0
        
        if self.reward_config.matchup_reward_weight > 0:
            step_reward += self._compute_matchup_quality(battle) * self.reward_config.matchup_reward_weight

        if self.reward_config.action_quality_weight > 0:
            step_reward += self._compute_action_quality(battle) * self.reward_config.action_quality_weight

        if battle.won:
            step_reward += self.reward_config.victory_reward
        elif battle.lost:
            step_reward += self.reward_config.defeat_penalty

        return (reward_delta + step_reward) * self.reward_config.reward_scale

    @staticmethod
    def _compute_matchup_quality(battle: AbstractBattle) -> float:
        """Stripped."""
        return 0.0

    def _compute_action_quality(self, battle: AbstractBattle) -> float:
        """
        Asymmetric move-quality shaping (see action_quality_weight in config).

        Offense: small reward only when the last action was a best-type damaging move.
        No penalty for suboptimal attacks (HP delta still punishes whiffs).

        Defense: small penalty when our active is weak to the opponent's best known
        damaging move. No reward for resisting (avoids pacifist stall exploits).
        """
        our = battle.active_pokemon
        opp = battle.opponent_active_pokemon
        if our is None or opp is None:
            return 0.0

        score = 0.0
        opp_types = tuple(opp.types or ())
        our_types = tuple(our.types or ())

        # --- Offensive: reward good attack only ---
        move_slot = _native_move_slot(self._last_native_action)
        if move_slot is not None and opp_types:
            known_moves = list(our.moves.values())
            if move_slot < len(known_moves):
                chosen_move = known_moves[move_slot]
                if _is_damaging_move(chosen_move):
                    chosen_eff = _move_type_effectiveness(chosen_move, opp_types)
                    best_eff = _best_damaging_effectiveness(our.moves, opp_types)
                    if best_eff > 0 and chosen_eff >= best_eff - 1e-6:
                        if chosen_eff >= 2.0:
                            score += 0.35
                        elif chosen_eff >= 1.0:
                            score += 0.15

        # --- Defensive: penalty for bad matchup only ---
        opp_moves = getattr(opp, "moves", None)
        if opp_moves and our_types:
            best_opp_eff = _best_damaging_effectiveness(opp_moves, our_types)
            if best_opp_eff >= 2.0:
                score -= 0.4
            elif best_opp_eff > 1.0:
                score -= 0.15

        return score
    
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
            self.reward_config.victory_reward
            if outcome == 1
            else self.reward_config.defeat_penalty
        )
        reward_hp_diff = hp_diff * self.reward_config.hp_value_weight
        reward_faint = (
            opp_fainted * self.reward_config.fainted_value
            + our_fainted * self.reward_config.fainted_penalty
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
            "opponent_type": self._opponent_context,
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
        """Convert a BattleOrder to action index natively."""
        try:
            # pylint: disable=no-member
            return SinglesEnv.order_to_action(order, battle, fake=fake, strict=True)
        except ValueError:
            if strict:
                raise

        self._fallback_events_current_episode += 1

        # Hard fallback: pick a random valid action from the mask
        # pylint: disable=no-member
        mask = SinglesEnv.get_action_mask(battle)
        valid_actions = [i for i, is_valid in enumerate(mask) if is_valid]
        if valid_actions:
            return np.int64(random.choice(valid_actions))
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
        opponent_team: Optional[str] = None,
        model_config_dict: Optional[Dict] = None,
        selfplay_weights_path: Optional[str] = None,
    ):
        super().__init__(env, opponent)
        self._battle_format = battle_format
        self._server_configuration = server_configuration
        self._opponent_team = opponent_team
        self._model_config_dict = model_config_dict
        self._selfplay_weights_path = selfplay_weights_path
        self._opponent_mix = self._normalize_opponent_mix(opponent_mix)
        self._opponent_pool: Dict[str, Any] = {}
        initial_key = self._opponent_key_from_instance(opponent)
        self._episode_total_actions = 0
        self._episode_switch_actions = 0
        self._episode_attack_actions = 0
        self._current_opponent_key = initial_key
        self._recent_action_stats: List[Dict[str, Any]] = []
        self._recent_observation_samples: List[Dict[str, Any]] = []
        self._recent_observation_cap = 64

        self._opponent_pool[initial_key] = opponent
        if hasattr(self.env, "set_opponent_context"):
            self.env.set_opponent_context(initial_key)
        if hasattr(self.env, "set_training_stage_context"):
            self.env.set_training_stage_context(0)
        self._stage_counter = 0

    @staticmethod
    def _normalize_opponent_mix(
        opponent_mix: Optional[Dict[str, float]],
    ) -> Dict[str, float]:
        default_mix = {"random": 1.0}
        if not opponent_mix:
            return default_mix

        valid = {}
        for key, val in opponent_mix.items():
            canonical = CurriculumSingleAgentWrapper._canonical_opponent_key(key)
            if canonical is not None and float(val) > 0:
                valid[canonical] = valid.get(canonical, 0.0) + float(val)

        total = sum(valid.values())
        if total <= 0:
            return default_mix
        return {k: v / total for k, v in valid.items()}

    def _choose_opponent_class(self):
        keys = list(self._opponent_mix.keys())
        weights = [self._opponent_mix[k] for k in keys]
        return random.choices(keys, weights=weights, k=1)[0]

    def _build_opponent(self, opponent_key: str):
        if opponent_key == "self":
            from src.training.self_play_player import SelfPlayPlayer

            opponent_id = f"self_{uuid.uuid4().hex[:6]}"
            opponent_config = AccountConfiguration(opponent_id, None)
            return SelfPlayPlayer(
                model_config_dict=self._model_config_dict or {},
                weights_path=self._selfplay_weights_path,
                battle_format=self._battle_format,
                account_configuration=opponent_config,
                server_configuration=self._server_configuration,
                team=self._opponent_team,
            )
        if opponent_key == "heuristic":
            opponent_class = SimpleHeuristicsPlayer
            opponent_id = f"hrs_{uuid.uuid4().hex[:6]}"
        elif opponent_key == "random_no_switch":
            from src.envs.random_no_switch_player import RandomNoSwitchPlayer

            opponent_class = RandomNoSwitchPlayer
            opponent_id = f"rndns_{uuid.uuid4().hex[:6]}"
        else:
            opponent_class = RandomPlayer
            opponent_id = f"rnd_{uuid.uuid4().hex[:6]}"
        opponent_config = AccountConfiguration(opponent_id, None)
        return opponent_class(
            battle_format=self._battle_format,
            account_configuration=opponent_config,
            server_configuration=self._server_configuration,
            team=self._opponent_team,
        )

    @staticmethod
    def _opponent_key_from_instance(opponent: Any) -> str:
        # Lazy import to avoid circular dependency at module load time.
        from src.training.self_play_player import SelfPlayPlayer

        if isinstance(opponent, SelfPlayPlayer):
            return "self"
        if isinstance(opponent, SimpleHeuristicsPlayer):
            return "heuristic"
        from src.envs.random_no_switch_player import RandomNoSwitchPlayer

        if isinstance(opponent, RandomNoSwitchPlayer):
            return "random_no_switch"
        return "random"

    @staticmethod
    def _canonical_opponent_key(value: Any) -> Optional[str]:
        if value is None:
            return None
        key = str(value).strip().lower()
        if not key or key == "unknown":
            return None
        if key == "heuristics":
            return "heuristic"
        key = re.sub(r"[^a-z0-9_.-]+", "_", key)
        key = re.sub(r"_+", "_", key).strip("_")
        return key or None

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
        self._current_opponent_key = opponent_key
        begin_episode = getattr(self.opponent, "begin_episode", None)
        if callable(begin_episode):
            begin_episode()
        if hasattr(self.env, "set_opponent_context"):
            self.env.set_opponent_context(opponent_key)
        self._episode_total_actions = 0
        self._episode_switch_actions = 0
        self._episode_attack_actions = 0
        self.env._last_native_action = -1
        if hasattr(self.env, "reset_tracking_state"):
            self.env.reset_tracking_state()
        if hasattr(self.env, "consume_fallback_events"):
            self.env.consume_fallback_events()
        result = super().reset(*args, **kwargs)
        obs = result[0] if isinstance(result, tuple) and len(result) > 0 else result
        self._record_observation_sample(obs)
        return result

    def step(self, action):
        if action is not None:
            action_int = int(action)
            self._last_action = action_int
            self.env._last_native_action = action_int
            self._episode_total_actions += 1
            if is_native_switch_action(action_int):
                self._episode_switch_actions += 1
            else:
                self._episode_attack_actions += 1

        step_action = np.int64(action) if action is not None else action
        result = super().step(step_action)
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
            action_stats: Dict[str, float] = {
                "episode_total_actions": float(self._episode_total_actions),
                "episode_switch_actions": float(self._episode_switch_actions),
                "episode_attack_actions": float(self._episode_attack_actions),
                "episode_fallback_events": float(fallback_events),
                "opponent_type": self._current_opponent_key,
            }
            if self._current_opponent_key == "self":
                sp = self._opponent_pool.get("self")
                pop_diag = getattr(sp, "pop_diagnostics", None)
                if callable(pop_diag):
                    diag = pop_diag()
                    infer_fb = float(diag.get("fallback_count", 0))
                    map_fb = float(diag.get("action_mapping_fallback_count", 0))
                    action_stats["selfplay_inference_fallback_turns"] = infer_fb
                    action_stats["selfplay_action_to_order_fallback_turns"] = map_fb
                    action_stats["selfplay_had_fallback"] = float(
                        infer_fb + map_fb > 0
                    )
            self._recent_action_stats.append(action_stats)
        obs = result[0] if isinstance(result, tuple) and len(result) > 0 else None
        self._record_observation_sample(obs)
        return result

    def _record_observation_sample(self, obs: Any) -> None:
        flat = flatten_agent_observation(obs)
        if flat is None:
            return
        required = {"obs", "species", "items", "abilities", "action_mask"}
        if not required.issubset(set(flat.keys())):
            return
        try:
            sample = {
                "obs": np.asarray(flat["obs"]).astype(np.float32, copy=False),
                "species": np.asarray(flat["species"]).astype(np.int64, copy=False),
                "items": np.asarray(flat["items"]).astype(np.int64, copy=False),
                "abilities": np.asarray(flat["abilities"]).astype(np.int64, copy=False),
                "action_mask": np.asarray(flat["action_mask"]).astype(
                    np.float32, copy=False
                ),
                "opponent_type": self._current_opponent_key,
                "training_stage_index": self._stage_counter,
            }
        except Exception:
            return
        self._recent_observation_samples.append(sample)
        if len(self._recent_observation_samples) > self._recent_observation_cap:
            self._recent_observation_samples = self._recent_observation_samples[
                -self._recent_observation_cap :
            ]

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
        self._stage_counter += 1
        if hasattr(self.env, "set_training_stage_context"):
            self.env.set_training_stage_context(self._stage_counter)

    def pop_recent_outcomes(self) -> List[int]:
        if hasattr(self.env, "pop_recent_outcomes"):
            return self.env.pop_recent_outcomes()
        return []

    def pop_recent_episode_stats(self) -> List[Dict[str, Any]]:
        env_stats = []
        if hasattr(self.env, "pop_recent_episode_stats"):
            env_stats = self.env.pop_recent_episode_stats()

        action_stats = self._recent_action_stats[:]
        self._recent_action_stats.clear()

        merged = []
        for idx, es in enumerate(env_stats):
            item = dict(es)
            if idx < len(action_stats):
                item.update(action_stats[idx])
            merged.append(item)
        return merged

    def pop_recent_observation_samples(
        self, max_samples: int = 3
    ) -> List[Dict[str, Any]]:
        max_samples = max(0, int(max_samples))
        if max_samples == 0:
            return []
        samples = self._recent_observation_samples[:max_samples]
        self._recent_observation_samples = self._recent_observation_samples[
            max_samples:
        ]
        return samples

    def pop_selfplay_diagnostics(self) -> Dict[str, Any]:
        """Collect diagnostics from the self-play opponent, if present."""
        if "self" not in self._opponent_pool:
            return {}
        sp = self._opponent_pool["self"]
        pop_fn = getattr(sp, "pop_diagnostics", None)
        if callable(pop_fn):
            return pop_fn()
        return {}

    def get_memory_counters(self) -> Dict[str, float]:
        out = {
            "wrapper_recent_action_stats_len": float(len(self._recent_action_stats)),
            "wrapper_recent_observation_samples_len": float(
                len(self._recent_observation_samples)
            ),
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
    reward += our_fainted * config.fainted_penalty

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
    player_team: Optional[str] = None,
    opponent_team: Optional[str] = None,
    model_config_dict: Optional[Dict] = None,
    selfplay_weights_path: Optional[str] = None,
):
    """
    Create an environment creator function for Ray RLlib.

    Args:
        battle_format: Battle format string
        server_host: Showdown server host
        server_port: Showdown server port
        reward_config: Reward configuration
        opponent_difficulty: "heuristic"/"heuristics", "random", or "random_no_switch"
        opponent_mix: Optional per-episode sampling mix, e.g.
            {"random": 0.7, "heuristic": 0.3} or {"random_no_switch": 1.0}
        player_team: Optional fixed Showdown team text for the learning agent
        opponent_team: Optional fixed Showdown team text for the opponent

    Returns:
        Callable that creates environments
    """

    def env_creator(env_config: Optional[Dict] = None):
        env_config = env_config or {}

        # Resolve settings
        fmt = env_config.get("battle_format", battle_format)
        host = env_config.get("server_host", server_host)
        rc = env_config.get("reward_config", reward_config or RewardConfig())
        difficulty = env_config.get("opponent_difficulty", opponent_difficulty)
        mix = env_config.get("opponent_mix", opponent_mix)
        if mix is None:
            mix = {difficulty: 1.0}
        p_team = env_config.get("player_team", player_team)
        o_team = env_config.get("opponent_team", opponent_team)

        if env_config.get("server_port") is not None:
            port = int(env_config["server_port"])
        else:
            num_srv = int(env_config.get("num_servers", 1))
            start_p = int(env_config.get("start_port", server_port))
            if num_srv <= 1:
                port = start_p
            else:
                wi = int(getattr(env_config, "worker_index", 0) or 0)
                nepw = int(env_config.get("num_envs_per_worker", 1))
                sub_i = int(env_config.get("_pokemon_sub_env_index", 0))
                env_config["_pokemon_sub_env_index"] = sub_i + 1
                slot = wi * nepw + sub_i
                port = start_p + (slot % num_srv)

        # Build proper websocket ServerConfiguration
        server_config = ServerConfiguration(
            f"ws://{host}:{port}/showdown/websocket",
            "https://play.pokemonshowdown.com/action.php?",
        )

        # Create a starting opponent. Wrapper will resample per episode
        # when opponent mixes are configured.
        opponent_id = f"rnd_{uuid.uuid4().hex[:6]}"
        if difficulty in {"heuristic", "heuristics"}:
            opponent_class = SimpleHeuristicsPlayer
            opponent_id = f"hrs_{uuid.uuid4().hex[:6]}"
        elif difficulty == "random_no_switch":
            from src.envs.random_no_switch_player import RandomNoSwitchPlayer

            opponent_class = RandomNoSwitchPlayer
            opponent_id = f"rndns_{uuid.uuid4().hex[:6]}"
        elif difficulty == "self":
            from src.training.self_play_player import SelfPlayPlayer

            opponent_class = SelfPlayPlayer
            opponent_id = f"self_{uuid.uuid4().hex[:6]}"
        else:
            opponent_class = RandomPlayer
        opponent_config = AccountConfiguration(opponent_id, None)
        env_opponent_id = f"{opponent_id}_e"
        env_opponent_config = AccountConfiguration(env_opponent_id, None)
        opponent_kwargs: Dict[str, Any] = {
            "battle_format": fmt,
            "account_configuration": opponent_config,
            "server_configuration": server_config,
            "team": o_team,
        }
        sp_path = env_config.get("selfplay_weights_path", selfplay_weights_path)
        m_cfg = env_config.get("model_config_dict", model_config_dict)
        if difficulty == "self":
            opponent_kwargs["model_config_dict"] = m_cfg or {}
            opponent_kwargs["weights_path"] = sp_path
        opponent = opponent_class(**opponent_kwargs)

        # Create the PettingZoo env
        player_id = f"RL_{uuid.uuid4().hex[:8]}"
        env = PokemonBattleEnv(
            reward_config=rc,
            battle_format=fmt,
            account_configuration1=AccountConfiguration(player_id, None),
            account_configuration2=env_opponent_config,
            server_configuration=server_config,
            strict=False,
            team=p_team,
        )

        # Wrap into single-agent gym env
        return CurriculumSingleAgentWrapper(
            env=env,
            opponent=opponent,
            battle_format=fmt,
            server_configuration=server_config,
            opponent_mix=mix,
            opponent_team=o_team,
            model_config_dict=model_config_dict,
            selfplay_weights_path=selfplay_weights_path,
        )

    return env_creator
