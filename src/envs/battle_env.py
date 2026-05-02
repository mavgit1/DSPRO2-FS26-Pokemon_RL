import gymnasium as gym
import json
import time
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

from src.action_space import (
    COMPRESSED_ACTION_SPACE_N,
    NATIVE_ACTION_SPACE_N,
    compressed_to_native_action,
    is_compressed_switch_action,
)
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
                shape=(COMPRESSED_ACTION_SPACE_N,),
                dtype=np.float32,
            ),
        }
    )


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
        return embed_battle(battle, opponent_type=self._opponent_context)

    def set_opponent_context(self, opponent_type: Optional[str]) -> None:
        """Attach selected opponent metadata to future observations."""
        self._opponent_context = opponent_type

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

        return self._compute_configured_delta_reward(battle)

    def _compute_configured_delta_reward(self, battle: AbstractBattle) -> float:
        """Poke-env style delta reward with asymmetric terminal values."""
        if battle not in self._reward_buffer:
            self._reward_buffer[battle] = 0.0

        current_value = 0.0
        hp_value = self.reward_config.hp_value_weight
        fainted_value = self.reward_config.fainted_value
        number_of_pokemons = 6

        for mon in battle.team.values():
            current_value += mon.current_hp_fraction * hp_value
            if mon.fainted:
                current_value -= fainted_value

        current_value += (number_of_pokemons - len(battle.team)) * hp_value

        for mon in battle.opponent_team.values():
            current_value -= mon.current_hp_fraction * hp_value
            if mon.fainted:
                current_value += fainted_value

        current_value -= (number_of_pokemons - len(battle.opponent_team)) * hp_value

        if battle.won:
            current_value += self.reward_config.victory_reward
        elif battle.lost:
            current_value += self.reward_config.defeat_penalty

        reward = current_value - self._reward_buffer[battle]
        self._reward_buffer[battle] = current_value
        return reward

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
            # #region agent log
            try:
                _b1 = getattr(self, "battle1", None)
                _b2 = getattr(self, "battle2", None)
                _which = (
                    "battle1"
                    if battle is _b1
                    else ("battle2" if battle is _b2 else "unknown")
                )
                _n = int(getattr(self, "_dbg_order_to_action_fail_n", 0)) + 1
                self._dbg_order_to_action_fail_n = _n
                if _n <= 50:
                    open(
                        "/var/home/tristan/CodingProjects/DSPRO2_Pokemon/DSPRO2-FS26-Pokemon_RL/.cursor/debug-a5a35e.log",
                        "a",
                        encoding="utf-8",
                    ).write(
                        json.dumps(
                            {
                                "sessionId": "a5a35e",
                                "runId": "post-fix",
                                "hypothesisId": "H1",
                                "location": "PokemonBattleEnv.order_to_action:ValueError",
                                "message": "strict_order_to_action_failed",
                                "data": {
                                    "which_battle": _which,
                                    "n": _n,
                                    "player_username": getattr(
                                        battle, "player_username", None
                                    ),
                                    "strict_param": bool(strict),
                                },
                                "timestamp": int(time.time() * 1000),
                            }
                        )
                        + "\n"
                    )
            except Exception:
                pass
            # #endregion
            if strict:
                raise

        # Retry with random legal-looking orders a fixed number of times.
        max_retries = 3
        for _ in range(max_retries):
            random_order = RandomPlayer.choose_random_singles_move(battle)
            try:
                # #region agent log
                try:
                    _nfb = int(getattr(self, "_dbg_fallback_retry_n", 0)) + 1
                    self._dbg_fallback_retry_n = _nfb
                    if _nfb <= 30:
                        _b1 = getattr(self, "battle1", None)
                        _b2 = getattr(self, "battle2", None)
                        _which = (
                            "battle1"
                            if battle is _b1
                            else ("battle2" if battle is _b2 else "unknown")
                        )
                        open(
                            "/var/home/tristan/CodingProjects/DSPRO2_Pokemon/DSPRO2-FS26-Pokemon_RL/.cursor/debug-a5a35e.log",
                            "a",
                            encoding="utf-8",
                        ).write(
                            json.dumps(
                                {
                                    "sessionId": "a5a35e",
                                    "runId": "post-fix",
                                    "hypothesisId": "H1",
                                    "location": "PokemonBattleEnv.order_to_action:fallback_retry",
                                    "message": "fallback_retry_increment",
                                    "data": {"which_battle": _which, "n": _nfb},
                                    "timestamp": int(time.time() * 1000),
                                }
                            )
                            + "\n"
                        )
                except Exception:
                    pass
                # #endregion
                self._fallback_events_current_episode += 1
                return SinglesEnv.order_to_action(
                    random_order, battle, fake=fake, strict=True
                )
            except ValueError:
                continue

        # Hard fallback: pick the first action that converts legally.
        for action in range(NATIVE_ACTION_SPACE_N):
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
        opponent_team: Optional[str] = None,
    ):
        super().__init__(env, opponent)
        self._battle_format = battle_format
        self._server_configuration = server_configuration
        self._opponent_team = opponent_team
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
        opponent_class = (
            SimpleHeuristicsPlayer if opponent_key == "heuristic" else RandomPlayer
        )
        opponent_id = f"Opp_{uuid.uuid4().hex[:6]}"
        opponent_config = AccountConfiguration(opponent_id, None)
        return opponent_class(
            battle_format=self._battle_format,
            account_configuration=opponent_config,
            server_configuration=self._server_configuration,
            team=self._opponent_team,
        )

    @staticmethod
    def _opponent_key_from_instance(opponent: Any) -> str:
        if isinstance(opponent, SimpleHeuristicsPlayer):
            return "heuristic"
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
        if hasattr(self.env, "set_opponent_context"):
            self.env.set_opponent_context(opponent_key)
        self._episode_total_actions = 0
        self._episode_switch_actions = 0
        self._episode_attack_actions = 0
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
            self._episode_total_actions += 1
            if is_compressed_switch_action(action_int):
                self._episode_switch_actions += 1
            else:
                self._episode_attack_actions += 1
            native_action = compressed_to_native_action(action_int, self.env.battle1)
            # #region agent log
            try:
                b1 = self.env.battle1
                if b1 is not None:
                    emb = self.env.embed_battle(b1)
                    m = np.asarray(emb["action_mask"], dtype=np.float32)
                    mk = float(m[action_int]) if 0 <= action_int < len(m) else -1.0
                    na_ok = True
                    try:
                        SinglesEnv.action_to_order(
                            native_action, b1, fake=False, strict=True
                        )
                    except Exception:
                        na_ok = False
                    _step_n = int(getattr(self, "_dbg_wrap_step_n", 0)) + 1
                    self._dbg_wrap_step_n = _step_n
                    _bad = mk < 0.5 or not na_ok
                    if _bad:
                        _bn = int(getattr(self, "_dbg_bad_rl_action_n", 0)) + 1
                        self._dbg_bad_rl_action_n = _bn
                        if _bn <= 40:
                            open(
                                "/var/home/tristan/CodingProjects/DSPRO2_Pokemon/DSPRO2-FS26-Pokemon_RL/.cursor/debug-a5a35e.log",
                                "a",
                                encoding="utf-8",
                            ).write(
                                json.dumps(
                                    {
                                        "sessionId": "a5a35e",
                                        "runId": "post-fix",
                                        "hypothesisId": "H2",
                                        "location": "CurriculumSingleAgentWrapper.step",
                                        "message": "rl_action_mask_or_native_mismatch",
                                        "data": {
                                            "action": action_int,
                                            "mask_value": mk,
                                            "native_verify_ok": na_ok,
                                            "opponent": self._current_opponent_key,
                                            "n": _bn,
                                        },
                                        "timestamp": int(time.time() * 1000),
                                    }
                                )
                                + "\n"
                            )
                    if _step_n % 300 == 0:
                        open(
                            "/var/home/tristan/CodingProjects/DSPRO2_Pokemon/DSPRO2-FS26-Pokemon_RL/.cursor/debug-a5a35e.log",
                            "a",
                            encoding="utf-8",
                        ).write(
                            json.dumps(
                                {
                                    "sessionId": "a5a35e",
                                    "runId": "post-fix",
                                    "hypothesisId": "H5",
                                    "location": "CurriculumSingleAgentWrapper.step",
                                    "message": "periodic_mask_health",
                                    "data": {
                                        "step_n": _step_n,
                                        "mask_sum": float(np.sum(m)),
                                        "mask_max": float(np.max(m)),
                                        "opponent": self._current_opponent_key,
                                        "last_action": action_int,
                                        "last_mask": mk,
                                    },
                                    "timestamp": int(time.time() * 1000),
                                }
                            )
                            + "\n"
                        )
            except Exception:
                pass
            # #endregion
        else:
            native_action = action

        result = super().step(native_action)
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
                    "opponent_type": self._current_opponent_key,
                }
            )
        obs = result[0] if isinstance(result, tuple) and len(result) > 0 else None
        self._record_observation_sample(obs)
        return result

    def _record_observation_sample(self, obs: Any) -> None:
        if not isinstance(obs, dict):
            return
        required = {"obs", "species", "items", "abilities", "action_mask"}
        if not required.issubset(set(obs.keys())):
            return
        try:
            sample = {
                "obs": np.asarray(obs["obs"]).astype(np.float32, copy=False),
                "species": np.asarray(obs["species"]).astype(np.int64, copy=False),
                "items": np.asarray(obs["items"]).astype(np.int64, copy=False),
                "abilities": np.asarray(obs["abilities"]).astype(np.int64, copy=False),
                "action_mask": np.asarray(obs["action_mask"]).astype(
                    np.float32, copy=False
                ),
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

        count = min(len(env_stats), len(action_stats))
        merged = []
        for idx in range(count):
            item = dict(env_stats[idx])
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
            opponent_id = f"hr_{uuid.uuid4().hex[:6]}"
        else:
            opponent_class = RandomPlayer
        opponent_config = AccountConfiguration(opponent_id, None)
        opponent = opponent_class(
            battle_format=fmt,
            account_configuration=opponent_config,
            server_configuration=server_config,
            team=o_team,
        )

        # Create the PettingZoo env
        player_id = f"RL_{uuid.uuid4().hex[:8]}"
        env = PokemonBattleEnv(
            reward_config=rc,
            battle_format=fmt,
            account_configuration1=AccountConfiguration(player_id, None),
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
        )

    return env_creator
