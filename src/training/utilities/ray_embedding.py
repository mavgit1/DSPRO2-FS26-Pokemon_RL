import numpy as np
import zlib
from gymnasium.spaces import Box, Dict, MultiDiscrete
from poke_env.environment.singles_env import SinglesEnv
from poke_env.battle.abstract_battle import AbstractBattle
from poke_env.battle.weather import Weather
from poke_env.battle.field import Field
from poke_env.battle.side_condition import SideCondition
from poke_env.battle.status import Status
from poke_env.battle.pokemon_type import PokemonType
from poke_env.battle.move_category import MoveCategory
from poke_env.battle.pokemon import Pokemon
from poke_env.battle.effect import Effect

# Caching enum lists for fast index lookups
WEATHER_LIST = list(Weather)
FIELD_LIST = list(Field)
SIDE_CONDITION_LIST = list(SideCondition)
STATUS_LIST = list(Status)
POKEMON_TYPE_LIST = list(PokemonType)
MOVE_CATEGORY_LIST = list(MoveCategory)
STATS = ['atk', 'def', 'spa', 'spd', 'spe', 'accuracy', 'evasion']
BASE_STATS = ['hp', 'atk', 'def', 'spa', 'spd', 'spe']

# Specific volatile effects we want to track
TRACKED_EFFECTS = [
    Effect.SUBSTITUTE, Effect.CONFUSION, Effect.TAUNT, Effect.ENCORE,
    Effect.LEECH_SEED, Effect.YAWN, Effect.PERISH1, Effect.PERISH2, Effect.PERISH3
]

def hash_str_to_int(s: str, max_val: int = 10000) -> int:
    """Helper to consistently hash string IDs (species, items, abilities) to a categorical integer."""
    if not s:
        return 0
    # Use adler32 for fast consistent hashing, map it to [1, max_val]
    return (zlib.adler32(s.encode('utf-8')) % max_val) + 1

class RayEmbeddingEnv(SinglesEnv):
    """
    A Gym environment that encapsulates a Transformer-friendly embedding of the Pokemon battle state.
    Observation is a Dict:
    - obs: (13, 163) floats (Token 0: Global, 1-6: Our Team, 7-12: Opponent Team)
    - species: (13,) ints (hash IDs)
    - items: (13,) ints (hash IDs)
    - abilities: (13,) ints (hash IDs)
    """

    NUM_TOKENS = 13
    TOKEN_DIM = 164
    MAX_ID_VAL = 20000

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.observation_spaces = {
            agent: Dict({
                "obs": Box(
                    low=-1.0, 
                    high=3.0,
                    shape=(self.NUM_TOKENS, self.TOKEN_DIM),
                    dtype=np.float32,
                ),
                # Categorical IDs for the 13 tokens (0 for padding/global token)
                "species": Box(low=0, high=self.MAX_ID_VAL, shape=(self.NUM_TOKENS,), dtype=np.int32),
                "items": Box(low=0, high=self.MAX_ID_VAL, shape=(self.NUM_TOKENS,), dtype=np.int32),
                "abilities": Box(low=0, high=self.MAX_ID_VAL, shape=(self.NUM_TOKENS,), dtype=np.int32),
            })
            for agent in self.possible_agents
        }

    def calc_reward(self, battle: AbstractBattle) -> float:
        return self.reward_computing_helper(
            battle, fainted_value=2.0, hp_value=1.0, victory_value=30.0
        )

    def embed_battle(self, battle: AbstractBattle) -> dict:
        obs = np.zeros((self.NUM_TOKENS, self.TOKEN_DIM), dtype=np.float32)
        species = np.zeros(self.NUM_TOKENS, dtype=np.int32)
        items = np.zeros(self.NUM_TOKENS, dtype=np.int32)
        abilities = np.zeros(self.NUM_TOKENS, dtype=np.int32)

        # -------------------------------------------------------------------------
        # Token 0: Global State
        # -------------------------------------------------------------------------
        global_idx = 0
        
        # Weather (9)
        if battle.weather and battle.weather in WEATHER_LIST:
            obs[0, global_idx + WEATHER_LIST.index(battle.weather)] = 1.0
        else:
            obs[0, global_idx + WEATHER_LIST.index(Weather.UNKNOWN)] = 1.0
        global_idx += len(WEATHER_LIST)

        # Fields (15)
        for field in battle.fields:
            if field in FIELD_LIST:
                obs[0, global_idx + FIELD_LIST.index(field)] = 1.0
        global_idx += len(FIELD_LIST)

        # Our Side Conditions (38)
        for side_cond in battle.side_conditions:
            if side_cond in SIDE_CONDITION_LIST:
                obs[0, global_idx + SIDE_CONDITION_LIST.index(side_cond)] = 1.0
        global_idx += len(SIDE_CONDITION_LIST)

        # Opponent Side Conditions (38)
        for side_cond in battle.opponent_side_conditions:
            if side_cond in SIDE_CONDITION_LIST:
                obs[0, global_idx + SIDE_CONDITION_LIST.index(side_cond)] = 1.0
        global_idx += len(SIDE_CONDITION_LIST)

        # -------------------------------------------------------------------------
        # Helper sequence for rendering teams into tokens
        # -------------------------------------------------------------------------
        def populate_pokemon_token(token_id: int, mon: Pokemon, is_active: bool, is_opponent: bool):
            idx = 0
            
            # 1. is_present
            obs[token_id, idx] = 1.0
            idx += 1
            
            # 2. is_active
            obs[token_id, idx] = 1.0 if is_active else 0.0
            idx += 1
            
            # 3. is_fainted
            obs[token_id, idx] = 1.0 if mon.fainted else 0.0
            idx += 1
            
            # 4. hp_fraction
            obs[token_id, idx] = mon.current_hp_fraction
            idx += 1
            
            # 5-10. Base Stats (Scaled down roughly by 200 for normalization)
            for stat_name in BASE_STATS:
                # `mon.base_stats` is a dict of strings. If unknown, we put 0
                val = mon.base_stats.get(stat_name, 0) if mon.base_stats else 0
                obs[token_id, idx] = val / 200.0
                idx += 1

            # 11-30. Types (Multi-hot, 20)
            if mon.type_1 and mon.type_1 in POKEMON_TYPE_LIST:
                obs[token_id, idx + POKEMON_TYPE_LIST.index(mon.type_1)] = 1.0
            if mon.type_2 and mon.type_2 in POKEMON_TYPE_LIST:
                obs[token_id, idx + POKEMON_TYPE_LIST.index(mon.type_2)] = 1.0
            idx += len(POKEMON_TYPE_LIST)

            # 31-37. Status (One-hot, 7)
            if mon.status and mon.status in STATUS_LIST:
                obs[token_id, idx + STATUS_LIST.index(mon.status)] = 1.0
            idx += len(STATUS_LIST)

            # 38-46. Tracked Volatile Effects (Substitute, Confusion, etc.) (9)
            if mon.effects:
                for effect_key, _ in mon.effects.items():
                    if effect_key in TRACKED_EFFECTS:
                        obs[token_id, idx + TRACKED_EFFECTS.index(effect_key)] = 1.0
            idx += len(TRACKED_EFFECTS)

            # 47-53. Stat Boosts (7)
            for stat in STATS:
                # Normalize boost from [-6, 6] to [-1, 1]
                val = mon.boosts.get(stat, 0) / 6.0
                obs[token_id, idx] = val
                idx += 1

            # 54. has_item
            obs[token_id, idx] = 1.0 if mon.item else 0.0
            idx += 1

            # 55. has_ability
            obs[token_id, idx] = 1.0 if mon.ability else 0.0
            idx += 1

            # 56. expected weight (mass affects things like Grass Knot)
            obs[token_id, idx] = mon.weight / 100.0 if mon.weight else 0.0
            idx += 1

            # 57-160. Moves (4 moves * 26 features = 104) -> exactly 160. (padding up to 164 for alignment if preferred, leaving exactly 160 utilized for Pokemon tokens)
            moves_iter = list(mon.moves.values())
            
            for m_i in range(4):
                if m_i < len(moves_iter):
                    move = moves_iter[m_i]
                    # move_present
                    obs[token_id, idx] = 1.0 
                    # base_power (scaled)
                    obs[token_id, idx + 1] = move.base_power / 100.0
                    # accuracy (scaled to 0-1 range)
                    obs[token_id, idx + 2] = move.accuracy if isinstance(move.accuracy, float) else (1.0 if move.accuracy is True else 0.0)
                    
                    # category (One-hot, 3)
                    if move.category in MOVE_CATEGORY_LIST:
                        obs[token_id, idx + 3 + MOVE_CATEGORY_LIST.index(move.category)] = 1.0
                    
                    # type (One-hot, 20)
                    if move.type in POKEMON_TYPE_LIST:
                        obs[token_id, idx + 6 + POKEMON_TYPE_LIST.index(move.type)] = 1.0
                idx += 26

            # Categorical Assignments
            species[token_id] = hash_str_to_int(mon.species) if mon.species else 0
            items[token_id] = hash_str_to_int(mon.item) if mon.item else 0
            abilities[token_id] = hash_str_to_int(mon.ability) if mon.ability else 0


        # -------------------------------------------------------------------------
        # Token 1-6: Our Team
        # -------------------------------------------------------------------------
        our_active = battle.active_pokemon
        if our_active:
            populate_pokemon_token(1, our_active, is_active=True, is_opponent=False)
        
        bench_idx = 2
        for mon in battle.team.values():
            if mon is not our_active and bench_idx <= 6:
                populate_pokemon_token(bench_idx, mon, is_active=False, is_opponent=False)
                bench_idx += 1

        # -------------------------------------------------------------------------
        # Token 7-12: Opponent Team
        # -------------------------------------------------------------------------
        opp_active = battle.opponent_active_pokemon
        if opp_active:
            populate_pokemon_token(7, opp_active, is_active=True, is_opponent=True)
        
        bench_idx = 8
        for mon in battle.opponent_team.values():
            if mon is not opp_active and bench_idx <= 12:
                populate_pokemon_token(bench_idx, mon, is_active=False, is_opponent=True)
                bench_idx += 1

        return {
            "obs": obs,
            "species": species,
            "items": items,
            "abilities": abilities
        }

