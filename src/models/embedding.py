import numpy as np
from typing import Dict, List, Optional, Any

# Poke-env imports (only for type hints and enums)
from poke_env.battle.abstract_battle import AbstractBattle
from poke_env.battle.pokemon import Pokemon
from poke_env.environment.singles_env import SinglesEnv
from poke_env.battle.weather import Weather
from poke_env.battle.field import Field
from poke_env.battle.side_condition import SideCondition
from poke_env.battle.status import Status
from poke_env.battle.pokemon_type import PokemonType
from poke_env.battle.move_category import MoveCategory
from poke_env.battle.effect import Effect

from src.models.vocab import get_embedding_vocab, vocab_sizes


# =============================================================================
# CACHED ENUM LISTS
# =============================================================================

WEATHER_LIST: List = list(Weather)
FIELD_LIST: List = list(Field)
SIDE_CONDITION_LIST: List = list(SideCondition)
STATUS_LIST: List = list(Status)
POKEMON_TYPE_LIST: List = list(PokemonType)
MOVE_CATEGORY_LIST: List = list(MoveCategory)

STATS = ['atk', 'def', 'spa', 'spd', 'spe', 'accuracy', 'evasion']
BASE_STATS = ['hp', 'atk', 'def', 'spa', 'spd', 'spe']

# Tracked volatile effects
TRACKED_EFFECTS = [
    Effect.SUBSTITUTE, Effect.CONFUSION, Effect.TAUNT, Effect.ENCORE,
    Effect.LEECH_SEED, Effect.YAWN, Effect.PERISH1, Effect.PERISH2, Effect.PERISH3
]


# =============================================================================
# EMBEDDING CONSTANTS
# =============================================================================

NUM_TOKENS = 13          # 1 global + 6 our team + 6 opponent team
TOKEN_DIM = 164
_VOCAB_SIZES = vocab_sizes()
SPECIES_VOCAB_SIZE = _VOCAB_SIZES["species_vocab_size"]
ITEM_VOCAB_SIZE = _VOCAB_SIZES["item_vocab_size"]
ABILITY_VOCAB_SIZE = _VOCAB_SIZES["ability_vocab_size"]
ACTION_SPACE_N = 22
NATIVE_SWITCH_ACTIONS = range(0, 6)
GLOBAL_EXTRA_FEATURE_NAMES = [
    "opponent_random",
    "opponent_heuristic",
    "opponent_other",
    "battle_turn_norm",
    "force_switch",
    "active_trapped",
    "available_move_count_norm",
    "available_switch_count_norm",
    "can_dynamax",
    "can_mega_evolve",
    "can_z_move",
]


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def _get_list_index(value: Any, lst: List) -> int:
    """Safely get index of value in list, return -1 if not found."""
    try:
        return lst.index(value)
    except ValueError:
        return -1


# =============================================================================
# POKEMON TOKEN EMBEDDING
# =============================================================================

def embed_pokemon(
    mon: Optional[Pokemon], 
    is_active: bool = False,
    is_opponent: bool = False
) -> Dict[str, Any]:
    """
    Embed a single Pokemon into a token vector.
    
    Args:
        mon: Pokemon object to embed (None for empty slot)
        is_active: Whether this is the active Pokemon
        is_opponent: Whether this is an opponent's Pokemon
    
    Returns:
        Dict with:
            - 'obs': np.ndarray of shape (TOKEN_DIM,)
            - 'species': int
            - 'items': int  
            - 'abilities': int
    """
    obs = np.zeros(TOKEN_DIM, dtype=np.float32)
    
    species_id = 0
    item_id = 0
    ability_id = 0
    
    # Handle empty slot
    if mon is None:
        return {
            'obs': obs,
            'species': species_id,
            'items': item_id,
            'abilities': ability_id,
        }
    
    idx = 0
    
    # ---------------------------------------------------------------------
    # 1. Presence flags (3 dims)
    # ---------------------------------------------------------------------
    obs[idx] = 1.0                      # is_present
    obs[idx + 1] = 1.0 if is_active else 0.0  # is_active
    obs[idx + 2] = 1.0 if mon.fainted else 0.0  # is_fainted
    idx += 3
    
    # ---------------------------------------------------------------------
    # 2. HP fraction (1 dim)
    # ---------------------------------------------------------------------
    obs[idx] = mon.current_hp_fraction
    idx += 1
    
    # ---------------------------------------------------------------------
    # 3. Base stats (6 dims, normalized by 200)
    # ---------------------------------------------------------------------
    if mon.base_stats:
        for stat_name in BASE_STATS:
            val = mon.base_stats.get(stat_name, 0)
            obs[idx] = val / 200.0
            idx += 1
    else:
        idx += 6
    obs[idx-6:idx] = np.nan_to_num(obs[idx-6:idx])  # Handle NaN
    
    # ---------------------------------------------------------------------
    # 4. Types (multi-hot, 20 dims)
    # ---------------------------------------------------------------------
    if mon.type_1:
        type_idx = _get_list_index(mon.type_1, POKEMON_TYPE_LIST)
        if type_idx >= 0:
            obs[idx + type_idx] = 1.0
    if mon.type_2:
        type_idx = _get_list_index(mon.type_2, POKEMON_TYPE_LIST)
        if type_idx >= 0:
            obs[idx + type_idx] = 1.0
    idx += len(POKEMON_TYPE_LIST)
    
    # ---------------------------------------------------------------------
    # 5. Status (one-hot, 7 dims)
    # ---------------------------------------------------------------------
    if mon.status:
        status_idx = _get_list_index(mon.status, STATUS_LIST)
        if status_idx >= 0:
            obs[idx + status_idx] = 1.0
    idx += len(STATUS_LIST)
    
    # ---------------------------------------------------------------------
    # 6. Tracked volatile effects (9 dims)
    # ---------------------------------------------------------------------
    if mon.effects:
        for effect_key in mon.effects:
            if effect_key in TRACKED_EFFECTS:
                obs[idx + TRACKED_EFFECTS.index(effect_key)] = 1.0
    idx += len(TRACKED_EFFECTS)
    
    # ---------------------------------------------------------------------
    # 7. Stat boosts (7 dims, normalized from [-6, 6] to [-1, 1])
    # ---------------------------------------------------------------------
    for stat in STATS:
        boost = mon.boosts.get(stat, 0) if mon.boosts else 0
        obs[idx] = boost / 6.0
        idx += 1
    
    # ---------------------------------------------------------------------
    # 8. Item/Ability flags (2 dims)
    # ---------------------------------------------------------------------
    obs[idx] = 1.0 if mon.item else 0.0      # has_item
    obs[idx + 1] = 1.0 if mon.ability else 0.0  # has_ability
    idx += 2
    
    # ---------------------------------------------------------------------
    # 9. Weight (1 dim, normalized by 100)
    # ---------------------------------------------------------------------
    obs[idx] = (mon.weight or 0) / 100.0
    idx += 1
    
    # ---------------------------------------------------------------------
    # 10. Moves (4 moves × 26 features = 104 dims)
    # ---------------------------------------------------------------------
    moves = list(mon.moves.values()) if mon.moves else []
    
    for m_i in range(4):
        if m_i < len(moves):
            move = moves[m_i]
            
            # Move present
            obs[idx] = 1.0
            
            # Base power (normalized by 100)
            obs[idx + 1] = (move.base_power or 0) / 100.0
            
            # Accuracy (0-1)
            if isinstance(move.accuracy, float):
                obs[idx + 2] = move.accuracy
            elif move.accuracy is True:
                obs[idx + 2] = 1.0
            else:
                obs[idx + 2] = 0.0
            
            # Category (one-hot, 3 dims)
            cat_idx = _get_list_index(move.category, MOVE_CATEGORY_LIST)
            if cat_idx >= 0:
                obs[idx + 3 + cat_idx] = 1.0
            
            # Type (one-hot, 20 dims)
            type_idx = _get_list_index(move.type, POKEMON_TYPE_LIST)
            if type_idx >= 0:
                obs[idx + 6 + type_idx] = 1.0
        
        idx += 26
    
    # ---------------------------------------------------------------------
    # 11. Categorical IDs
    # ---------------------------------------------------------------------
    vocab = get_embedding_vocab()
    species_id = vocab.species_id(mon.species)
    item_id = vocab.item_id(mon.item)
    ability_id = vocab.ability_id(mon.ability)
    
    return {
        'obs': obs,
        'species': species_id,
        'items': item_id,
        'abilities': ability_id,
    }


# =============================================================================
# FULL BATTLE EMBEDDING
# =============================================================================

def embed_battle(
    battle: AbstractBattle,
    opponent_type: Optional[str] = None,
) -> Dict[str, np.ndarray]:
    """
    Convert full battle state to transformer-ready embedding.
    
    Output structure:
        - Token 0: Global state (weather, fields, side conditions)
        - Tokens 1-6: Our team (token 1 = active)
        - Tokens 7-12: Opponent team (token 7 = active)
    
    Args:
        battle: AbstractBattle object from poke-env
        opponent_type: Optional selected opponent label for the episode.
    
    Returns:
        Dict with:
            - 'obs': np.ndarray of shape (NUM_TOKENS, TOKEN_DIM)
            - 'species': np.ndarray of shape (NUM_TOKENS,)
            - 'items': np.ndarray of shape (NUM_TOKENS,)
            - 'abilities': np.ndarray of shape (NUM_TOKENS,)
            - 'action_mask': np.ndarray of shape (num_actions,)
    """
    obs = np.zeros((NUM_TOKENS, TOKEN_DIM), dtype=np.float32)
    species = np.zeros(NUM_TOKENS, dtype=np.int32)
    items = np.zeros(NUM_TOKENS, dtype=np.int32)
    abilities = np.zeros(NUM_TOKENS, dtype=np.int32)
    
    # -------------------------------------------------------------------------
    # Token 0: Global State
    # -------------------------------------------------------------------------
    global_idx = 0
    
    # Weather (9 dims)
    if battle.weather:
        weather_idx = _get_list_index(battle.weather, WEATHER_LIST)
        if weather_idx >= 0:
            obs[0, global_idx + weather_idx] = 1.0
    global_idx += len(WEATHER_LIST)
    
    # Fields/Terrain (15 dims)
    for field in battle.fields:
        field_idx = _get_list_index(field, FIELD_LIST)
        if field_idx >= 0:
            obs[0, global_idx + field_idx] = 1.0
    global_idx += len(FIELD_LIST)
    
    # Our side conditions (38 dims)
    for sc in battle.side_conditions:
        sc_idx = _get_list_index(sc, SIDE_CONDITION_LIST)
        if sc_idx >= 0:
            obs[0, global_idx + sc_idx] = 1.0
    global_idx += len(SIDE_CONDITION_LIST)
    
    # Opponent side conditions (38 dims)
    for sc in battle.opponent_side_conditions:
        sc_idx = _get_list_index(sc, SIDE_CONDITION_LIST)
        if sc_idx >= 0:
            obs[0, global_idx + sc_idx] = 1.0
    global_idx += len(SIDE_CONDITION_LIST)

    # Extra global context features. These fit in the existing spare token
    # capacity, so improves coverage without changing TOKEN_DIM.
    extra_features = _global_extra_features(battle, opponent_type)
    available = max(0, TOKEN_DIM - global_idx)
    if available > 0:
        count = min(len(extra_features), available)
        obs[0, global_idx : global_idx + count] = extra_features[:count]
    
    # -------------------------------------------------------------------------
    # Tokens 1-6: Our Team
    # -------------------------------------------------------------------------
    our_active = battle.active_pokemon
    if our_active:
        token_data = embed_pokemon(our_active, is_active=True, is_opponent=False)
        obs[1] = token_data['obs']
        species[1] = token_data['species']
        items[1] = token_data['items']
        abilities[1] = token_data['abilities']
    
    bench_idx = 2
    for mon in battle.team.values():
        if mon is not our_active and bench_idx <= 6:
            token_data = embed_pokemon(mon, is_active=False, is_opponent=False)
            obs[bench_idx] = token_data['obs']
            species[bench_idx] = token_data['species']
            items[bench_idx] = token_data['items']
            abilities[bench_idx] = token_data['abilities']
            bench_idx += 1
    
    # -------------------------------------------------------------------------
    # Tokens 7-12: Opponent Team
    # -------------------------------------------------------------------------
    opp_active = battle.opponent_active_pokemon
    if opp_active:
        token_data = embed_pokemon(opp_active, is_active=True, is_opponent=True)
        obs[7] = token_data['obs']
        species[7] = token_data['species']
        items[7] = token_data['items']
        abilities[7] = token_data['abilities']
    
    opp_bench_idx = 8
    for mon in battle.opponent_team.values():
        if mon is not opp_active and opp_bench_idx <= 12:
            token_data = embed_pokemon(mon, is_active=False, is_opponent=True)
            obs[opp_bench_idx] = token_data['obs']
            species[opp_bench_idx] = token_data['species']
            items[opp_bench_idx] = token_data['items']
            abilities[opp_bench_idx] = token_data['abilities']
            opp_bench_idx += 1
    
    # -------------------------------------------------------------------------
    # Action Mask
    # -------------------------------------------------------------------------
    action_mask = get_action_mask(battle)
    
    return {
        'obs': obs,
        'species': species,
        'items': items,
        'abilities': abilities,
        'action_mask': action_mask,
    }


def _global_extra_features(
    battle: AbstractBattle,
    opponent_type: Optional[str],
) -> np.ndarray:
    features = np.zeros(len(GLOBAL_EXTRA_FEATURE_NAMES), dtype=np.float32)
    opponent_key = _canonical_opponent_type(opponent_type)
    if opponent_key == "random":
        features[0] = 1.0
    elif opponent_key == "heuristic":
        features[1] = 1.0
    elif opponent_key:
        features[2] = 1.0

    features[3] = min(float(max(0, int(getattr(battle, "turn", 0)))) / 100.0, 1.0)
    features[4] = 1.0 if bool(getattr(battle, "force_switch", False)) else 0.0

    active = getattr(battle, "active_pokemon", None)
    features[5] = 1.0 if active is not None and bool(getattr(active, "trapped", False)) else 0.0
    features[6] = min(float(len(getattr(battle, "available_moves", []) or [])) / 4.0, 1.0)
    features[7] = min(float(len(getattr(battle, "available_switches", []) or [])) / 6.0, 1.0)
    features[8] = 1.0 if bool(getattr(battle, "can_dynamax", False)) else 0.0
    features[9] = 1.0 if bool(getattr(battle, "can_mega_evolve", False)) else 0.0
    features[10] = 1.0 if bool(getattr(battle, "can_z_move", False)) else 0.0
    return features


def _canonical_opponent_type(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    key = str(value).strip().lower()
    if not key or key == "unknown":
        return None
    if key == "heuristics":
        return "heuristic"
    return key


# =============================================================================
# ACTION MASKING
# =============================================================================

def get_action_mask(battle: AbstractBattle) -> np.ndarray:
    """
    Generate action mask for valid actions.
    
    Action space layout (gen8randombattle - 22 native poke-env actions):
        - 0-5: Switches
        - 6-9: Moves
        - 10-13: Mega Evolution
        - 14-17: Z-Move
        - 18-21: Dynamax
    
    Args:
        battle: AbstractBattle object
    
    Returns:
        np.ndarray of shape (22,) with 1.0 for valid actions, 0.0 for invalid
    """
    # For gen8 singles the action space is 22:
    # 0-5 switches, 6-9 moves, 10-13 mega, 14-17 z-move, 18-21 dynamax.
    # Use poke-env's own converter so mask indices exactly match env semantics.
    mask = np.zeros(ACTION_SPACE_N, dtype=np.float32)

    for action in range(ACTION_SPACE_N):
        try:
            SinglesEnv.action_to_order(
                np.int64(action),
                battle,
                fake=False,
                strict=True,
            )
            mask[action] = 1.0
        except (IndexError, ValueError):
            continue

    # Safety fallback for rare edge cases (prevents all-zero mask instability).
    if not mask.any():
        mask[6] = 1.0

    return mask


def get_valid_action_indices(battle: AbstractBattle) -> List[int]:
    """Get list of valid action indices."""
    return [i for i, valid in enumerate(get_action_mask(battle)) if valid]


def is_native_switch_action(action: int) -> bool:
    """Return whether a native gen8 singles poke-env action is a switch."""
    return int(action) in NATIVE_SWITCH_ACTIONS


# =============================================================================
# WIN PROBABILITY ESTIMATION (todo: Replace with a trained value network later)
# =============================================================================

def estimate_win_probability(battle: AbstractBattle) -> float:
    """
    Estimate win probability using heuristics.
    
    For better accuracy, use a trained value network instead.
    
    Args:
        battle: AbstractBattle object
    
    Returns:
        Float in [0, 1] representing estimated win probability
    """
    if battle.won:
        return 1.0
    if battle.lost:
        return 0.0
    
    our_score = 0.0
    opp_score = 0.0
    
    # Pokemon count (alive vs fainted)
    our_alive = sum(1 for m in battle.team.values() if not m.fainted)
    opp_alive = sum(1 for m in battle.opponent_team.values() if not m.fainted)
    our_score += our_alive * 15
    opp_score += opp_alive * 15
    
    # HP totals
    for mon in battle.team.values():
        if not mon.fainted:
            our_score += mon.current_hp_fraction * 10
    
    for mon in battle.opponent_team.values():
        if not mon.fainted:
            opp_score += mon.current_hp_fraction * 10
    
    # Boosts on active Pokemon
    if battle.active_pokemon and battle.active_pokemon.boosts:
        for boost in battle.active_pokemon.boosts.values():
            our_score += boost * 2
    
    # Normalize to probability
    total = our_score + opp_score
    if total <= 0:
        return 0.5
    
    return our_score / total