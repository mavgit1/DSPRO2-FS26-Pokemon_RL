import numpy as np
from gymnasium.spaces import Box
from poke_env.environment.singles_env import SinglesEnv
from poke_env.battle.abstract_battle import AbstractBattle
from poke_env.battle.weather import Weather
from poke_env.battle.field import Field
from poke_env.battle.side_condition import SideCondition
from poke_env.battle.status import Status
from poke_env.battle.pokemon_type import PokemonType
from poke_env.battle.move_category import MoveCategory
from poke_env.battle.pokemon import Pokemon

# Caching enum lists for fast index lookups
WEATHER_LIST = list(Weather)
FIELD_LIST = list(Field)
SIDE_CONDITION_LIST = list(SideCondition)
STATUS_LIST = list(Status)
POKEMON_TYPE_LIST = list(PokemonType)
MOVE_CATEGORY_LIST = list(MoveCategory)
STATS = ['atk', 'def', 'spa', 'spd', 'spe', 'accuracy', 'evasion']

class RayEmbeddingEnv(SinglesEnv):
    """
    A Gym environment that encapsulates a Transformer-friendly embedding of the Pokemon battle state.
    The observation is a 2D array of shape (13, 144):
    - Token 0: Global battle state (Weather, Fields, Side Conditions) zero-padded.
    - Tokens 1-6: Our team's Pokemon (Active Pokemon is always Token 1, followed by bench).
    - Tokens 7-12: Opponent team's Pokemon (Active is Token 7, followed by bench).
    """

    NUM_TOKENS = 13
    TOKEN_DIM = 144

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.observation_spaces = {
            agent: Box(
                low=-1.0, 
                high=3.0, # Most are 0-1, some bases/multipliers might go slightly above 1
                shape=(self.NUM_TOKENS, self.TOKEN_DIM),
                dtype=np.float32,
            )
            for agent in self.possible_agents
        }

    def calc_reward(self, battle: AbstractBattle) -> float:
        return self.reward_computing_helper(
            battle, fainted_value=2.0, hp_value=1.0, victory_value=30.0
        )

    def embed_battle(self, battle: AbstractBattle) -> np.ndarray:
        obs = np.zeros((self.NUM_TOKENS, self.TOKEN_DIM), dtype=np.float32)

        # Token 0: Global State
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

        # Our Side Conditions (24)
        for side_cond in battle.side_conditions:
            if side_cond in SIDE_CONDITION_LIST:
                obs[0, global_idx + SIDE_CONDITION_LIST.index(side_cond)] = 1.0
        global_idx += len(SIDE_CONDITION_LIST)

        # Opponent Side Conditions (24)
        for side_cond in battle.opponent_side_conditions:
            if side_cond in SIDE_CONDITION_LIST:
                obs[0, global_idx + SIDE_CONDITION_LIST.index(side_cond)] = 1.0
        global_idx += len(SIDE_CONDITION_LIST)

        # Helper sequence for rendering teams into tokens
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
            
            # 5-24. Types (Multi-hot, 20)
            if mon.type_1 and mon.type_1 in POKEMON_TYPE_LIST:
                obs[token_id, idx + POKEMON_TYPE_LIST.index(mon.type_1)] = 1.0
            if mon.type_2 and mon.type_2 in POKEMON_TYPE_LIST:
                obs[token_id, idx + POKEMON_TYPE_LIST.index(mon.type_2)] = 1.0
            idx += len(POKEMON_TYPE_LIST)

            # 25-31. Status (One-hot, 7)
            if mon.status and mon.status in STATUS_LIST:
                obs[token_id, idx + STATUS_LIST.index(mon.status)] = 1.0
            idx += len(STATUS_LIST)

            # 32-38. Boosts (7)
            for stat in STATS:
                # Normalize boost from [-6, 6] to [-1, 1]
                val = mon.boosts.get(stat, 0) / 6.0
                obs[token_id, idx] = val
                idx += 1

            # 39. has_item
            obs[token_id, idx] = 1.0 if mon.item else 0.0
            idx += 1

            # 40. has_ability
            obs[token_id, idx] = 1.0 if mon.ability else 0.0
            idx += 1

            # 41-144. Moves (4 moves * 26 features = 104)
            moves_iter = list(mon.moves.values())
            
            # If active, battle.available_moves contains the actual accessible move objects 
            # (which handles PP/Encored/Taunted etc.)
            # But the opponent's available_moves is not known. We stick to mon.moves
            
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

        # -------------------------------------------------------------------------
        # Token 1-6: Our Team
        # -------------------------------------------------------------------------
        # Always place active pokemon at Token 1 for consistency
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

        return obs
