import numpy as np
from typing import Dict, List, Optional, Any

from poke_env.battle.abstract_battle import AbstractBattle
from poke_env.battle.pokemon import Pokemon
from poke_env.battle.weather import Weather
from poke_env.battle.field import Field
from poke_env.battle.side_condition import SideCondition
from poke_env.battle.status import Status
from poke_env.battle.pokemon_type import PokemonType
from poke_env.battle.move_category import MoveCategory
from poke_env.battle.effect import Effect
from poke_env.environment.singles_env import SinglesEnv

from src.action_space import NATIVE_ACTION_SPACE_N
from src.models.feature_tables import ITEM_CLASS_NAMES, item_class_vector
from src.models.matchup import best_damage_frac, matchup_vector, speed_order, type_multiplier
from src.models.role_scores import ROLE_SCORE_NAMES, compute_role_scores, estimate_stats
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
MOVE_SLOTS = 4
PRESENCE_DIM = 3
HP_DIM = 1
BASE_STAT_DIM = 6
TYPE_DIM = len(POKEMON_TYPE_LIST)
STATUS_DIM = len(STATUS_LIST)
VOLATILE_DIM = len(TRACKED_EFFECTS)
BOOST_DIM = 7
ITEM_ABILITY_FLAG_DIM = 2
WEIGHT_DIM = 1
MOVE_SLOT_DENSE_DIM = 3 + len(MOVE_CATEGORY_LIST) + TYPE_DIM  # present, bp, acc, category, type
MOVE_DENSE_DIM = MOVE_SLOTS * MOVE_SLOT_DENSE_DIM
VISIBILITY_DIM = 4
LEGACY_POKEMON_DIM = (
    PRESENCE_DIM
    + HP_DIM
    + BASE_STAT_DIM
    + TYPE_DIM
    + STATUS_DIM
    + VOLATILE_DIM
    + BOOST_DIM
    + ITEM_ABILITY_FLAG_DIM
    + WEIGHT_DIM
    + MOVE_DENSE_DIM
    + VISIBILITY_DIM
)
KINEMATIC_DIM = 23        # level + 6 stats + 4 moves × (priority, pp, stab, contact)
ROLE_DIM = len(ROLE_SCORE_NAMES)
ITEM_CLASS_DIM = len(ITEM_CLASS_NAMES)
MATCHUP_DIM = 5           # speed_order, spe_ratio, off, def, tank_vs_active
TOKEN_DIM = (
    LEGACY_POKEMON_DIM + KINEMATIC_DIM + ROLE_DIM + ITEM_CLASS_DIM + MATCHUP_DIM
)
KINEMATIC_START = LEGACY_POKEMON_DIM
ROLE_START = KINEMATIC_START + KINEMATIC_DIM
ITEM_CLASS_START = ROLE_START + ROLE_DIM
MATCHUP_START = ITEM_CLASS_START + ITEM_CLASS_DIM
_VOCAB_SIZES = vocab_sizes()
SPECIES_VOCAB_SIZE = _VOCAB_SIZES["species_vocab_size"]
ITEM_VOCAB_SIZE = _VOCAB_SIZES["item_vocab_size"]
ABILITY_VOCAB_SIZE = _VOCAB_SIZES["ability_vocab_size"]
MOVE_VOCAB_SIZE = _VOCAB_SIZES["move_vocab_size"]
ACTION_SPACE_N = NATIVE_ACTION_SPACE_N
GLOBAL_EXTRA_START_IDX = (
    len(WEATHER_LIST) + len(FIELD_LIST) + 2 * len(SIDE_CONDITION_LIST)
)
GLOBAL_EXTRA_FEATURE_NAMES = [
    "opponent_random",
    "opponent_random_no_switch",
    "opponent_heuristic",
    "opponent_self",
    "opponent_historical",
    "opponent_other",
    "training_stage_index",
    "battle_turn_norm",
    "force_switch",
    "active_trapped",
    "available_move_count_norm",
    "available_switch_count_norm",
    "can_dynamax",
    "can_mega_evolve",
    "can_z_move",
    "move_1_multiplier",
    "move_2_multiplier",
    "move_3_multiplier",
    "move_4_multiplier",
    "opp_move_1_multiplier",
    "opp_move_2_multiplier",
    "opp_move_3_multiplier",
    "opp_move_4_multiplier",
    "we_outspeed",
    "last_damage_dealt_frac",
    "last_damage_taken_frac",
    "ko_likely",
    "two_hko_likely",
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

def _infer_last_move_id(mon: Pokemon, vocab) -> int:
    """Best-effort last-used move from PP deltas on the active Pokemon."""
    if not mon or not mon.moves:
        return 0
    candidates = []
    for move in mon.moves.values():
        try:
            if move.current_pp < move.max_pp:
                candidates.append((move.current_pp / max(move.max_pp, 1), move))
        except Exception:
            continue
    if not candidates:
        return 0
    candidates.sort(key=lambda pair: pair[0])
    return vocab.move_id(candidates[0][1])


def _species_is_revealed(mon: Pokemon, is_opponent: bool) -> bool:
    if mon is None:
        return False
    if not is_opponent:
        return True
    species = getattr(mon, "species", None) or getattr(mon, "name", None)
    if not species:
        return False
    normalized = str(species).strip().lower()
    return normalized not in {"", "unknown", "unknown_pokemon", "?"}


def embed_pokemon(
    mon: Optional[Pokemon], 
    is_active: bool = False,
    is_opponent: bool = False,
    vs_mon: Optional[Pokemon] = None,
) -> Dict[str, Any]:
    """
    Embed a single Pokemon into a token vector.
    
    Args:
        mon: Pokemon object to embed (None for empty slot)
        is_active: Whether this is the active Pokemon
        is_opponent: Whether this is an opponent's Pokemon
        vs_mon: Opposing active Pokemon for speed/matchup features
    
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
    move_ids = np.zeros(MOVE_SLOTS, dtype=np.int32)
    last_move_id = 0
    
    # Handle empty slot
    if mon is None:
        obs[0] = 1.0  # is_empty_slot
        return {
            'obs': obs,
            'species': species_id,
            'items': item_id,
            'abilities': ability_id,
            'moves': move_ids,
            'last_move': last_move_id,
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
            
            type_idx = _get_list_index(move.type, POKEMON_TYPE_LIST)
            if type_idx >= 0:
                obs[idx + 6 + type_idx] = 1.0
        
        idx += MOVE_SLOT_DENSE_DIM
    
    # ---------------------------------------------------------------------
    # 11. Visibility / opponent-transparency flags (4 dims)
    # ---------------------------------------------------------------------
    vocab = get_embedding_vocab()
    species_revealed = _species_is_revealed(mon, is_opponent)
    known_moves = list(mon.moves.values()) if mon.moves else []
    obs[idx] = 0.0  # is_empty_slot
    obs[idx + 1] = 1.0 if species_revealed else 0.0
    obs[idx + 2] = min(len(known_moves) / float(MOVE_SLOTS), 1.0)
    obs[idx + 3] = 1.0 if (is_opponent and not species_revealed) else 0.0
    idx += 4

    # ---------------------------------------------------------------------
    # 12. Kinematics: level, estimated stats, per-move extras
    # ---------------------------------------------------------------------
    stats = estimate_stats(mon) if species_revealed else {}
    level = int(getattr(mon, "level", 0) or 0) if species_revealed else 0
    obs[idx] = min(max(level, 0) / 100.0, 1.0)
    for offset, name in enumerate(BASE_STATS):
        obs[idx + 1 + offset] = min(float(stats.get(name, 0.0)) / 400.0, 1.0)
    idx += 7

    our_types = tuple(t for t in (getattr(mon, "types", None) or ()) if t is not None)
    if not our_types:
        our_types = tuple(
            t for t in (getattr(mon, "type_1", None), getattr(mon, "type_2", None)) if t is not None
        )
    for m_i in range(MOVE_SLOTS):
        slot = idx + m_i * 4
        if m_i < len(known_moves):
            move = known_moves[m_i]
            try:
                priority = float(getattr(move, "priority", 0) or 0)
            except (TypeError, ValueError):
                priority = 0.0
            obs[slot] = float(np.clip(priority / 7.0, -1.0, 1.0))
            max_pp = float(getattr(move, "max_pp", 0) or 0)
            current_pp = float(getattr(move, "current_pp", 0) or 0)
            obs[slot + 1] = current_pp / max_pp if max_pp > 0 else 0.0
            move_type = getattr(move, "type", None)
            obs[slot + 2] = 1.0 if (move_type is not None and move_type in our_types) else 0.0
            flags = getattr(move, "flags", None) or ()
            flag_names = {str(f).lower() for f in flags}
            obs[slot + 3] = 1.0 if "contact" in flag_names else 0.0
    idx += 16

    # ---------------------------------------------------------------------
    # 13. Fractional roles + item class
    # ---------------------------------------------------------------------
    if species_revealed:
        obs[idx : idx + ROLE_DIM] = compute_role_scores(
            mon,
            is_opponent=is_opponent,
            revealed_move_frac=min(len(known_moves) / float(MOVE_SLOTS), 1.0),
        )
    idx += ROLE_DIM
    if species_revealed or not is_opponent:
        obs[idx : idx + ITEM_CLASS_DIM] = np.asarray(
            item_class_vector(getattr(mon, "item", None)), dtype=np.float32
        )
    idx += ITEM_CLASS_DIM

    # ---------------------------------------------------------------------
    # 14. Matchup vs the opposing active
    # ---------------------------------------------------------------------
    if species_revealed and vs_mon is not None:
        obs[idx : idx + MATCHUP_DIM] = matchup_vector(mon, vs_mon)
    idx += MATCHUP_DIM
    if idx != TOKEN_DIM:
        raise RuntimeError(f"pokemon token packed {idx} dims, expected {TOKEN_DIM}")

    # ---------------------------------------------------------------------
    # 15. Categorical IDs
    # ---------------------------------------------------------------------
    species_id = vocab.species_id(mon.species) if species_revealed else 0
    item_id = vocab.item_id(mon.item) if species_revealed or not is_opponent else 0
    ability_id = vocab.ability_id(mon.ability) if species_revealed or not is_opponent else 0

    for m_i in range(MOVE_SLOTS):
        if m_i < len(known_moves):
            move_ids[m_i] = vocab.move_id(known_moves[m_i])

    if is_active:
        last_move_id = _infer_last_move_id(mon, vocab)
    
    return {
        'obs': obs,
        'species': species_id,
        'items': item_id,
        'abilities': ability_id,
        'moves': move_ids,
        'last_move': last_move_id,
    }


# =============================================================================
# FULL BATTLE EMBEDDING
# =============================================================================

def embed_battle(
    battle: AbstractBattle,
    opponent_type: Optional[str] = None,
    training_stage_index: Optional[int] = None,
    last_damage_dealt: float = 0.0,
    last_damage_taken: float = 0.0,
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
        training_stage_index: Optional curriculum stage index counter.
    
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
    moves = np.zeros((NUM_TOKENS, MOVE_SLOTS), dtype=np.int32)
    last_move = np.zeros(NUM_TOKENS, dtype=np.int32)
    
    # -------------------------------------------------------------------------
    # Token 0: Global State
    # -------------------------------------------------------------------------
    global_idx = 0
    
    # Weather (9 dims) — battle.weather is Dict[Weather, int], iterate keys
    for weather in battle.weather:
        weather_idx = _get_list_index(weather, WEATHER_LIST)
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
    extra_features = _global_extra_features(
        battle,
        opponent_type,
        training_stage_index=training_stage_index,
        last_damage_dealt=last_damage_dealt,
        last_damage_taken=last_damage_taken,
    )
    available = max(0, TOKEN_DIM - global_idx)
    if available > 0:
        count = min(len(extra_features), available)
        obs[0, global_idx : global_idx + count] = extra_features[:count]
    
    # -------------------------------------------------------------------------
    # Tokens 1-6: Our Team
    # -------------------------------------------------------------------------
    our_active = battle.active_pokemon
    opp_active = battle.opponent_active_pokemon
    if our_active:
        token_data = embed_pokemon(
            our_active, is_active=True, is_opponent=False, vs_mon=opp_active
        )
        obs[1] = token_data['obs']
        species[1] = token_data['species']
        items[1] = token_data['items']
        abilities[1] = token_data['abilities']
        moves[1] = token_data['moves']
        last_move[1] = token_data['last_move']
    
    bench_idx = 2
    for mon in battle.team.values():
        if mon is not our_active and bench_idx <= 6:
            token_data = embed_pokemon(
                mon, is_active=False, is_opponent=False, vs_mon=opp_active
            )
            obs[bench_idx] = token_data['obs']
            species[bench_idx] = token_data['species']
            items[bench_idx] = token_data['items']
            abilities[bench_idx] = token_data['abilities']
            moves[bench_idx] = token_data['moves']
            last_move[bench_idx] = token_data['last_move']
            bench_idx += 1
    
    # -------------------------------------------------------------------------
    # Tokens 7-12: Opponent Team
    # -------------------------------------------------------------------------
    if opp_active:
        token_data = embed_pokemon(
            opp_active, is_active=True, is_opponent=True, vs_mon=our_active
        )
        obs[7] = token_data['obs']
        species[7] = token_data['species']
        items[7] = token_data['items']
        abilities[7] = token_data['abilities']
        moves[7] = token_data['moves']
        last_move[7] = token_data['last_move']
    
    opp_bench_idx = 8
    for mon in battle.opponent_team.values():
        if mon is not opp_active and opp_bench_idx <= 12:
            token_data = embed_pokemon(
                mon, is_active=False, is_opponent=True, vs_mon=our_active
            )
            obs[opp_bench_idx] = token_data['obs']
            species[opp_bench_idx] = token_data['species']
            items[opp_bench_idx] = token_data['items']
            abilities[opp_bench_idx] = token_data['abilities']
            moves[opp_bench_idx] = token_data['moves']
            last_move[opp_bench_idx] = token_data['last_move']
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
        'moves': moves,
        'last_move': last_move,
        'action_mask': action_mask,
    }


def _global_extra_features(
    battle: AbstractBattle,
    opponent_type: Optional[str],
    training_stage_index: Optional[int] = None,
    last_damage_dealt: float = 0.0,
    last_damage_taken: float = 0.0,
) -> np.ndarray:
    features = np.zeros(len(GLOBAL_EXTRA_FEATURE_NAMES), dtype=np.float32)
    # Opponent-type oracle bits intentionally disabled (indices 0-5 stay zero).
    # Real opponent identity must be inferred from revealed battle state only.

    features[6] = float(max(0, int(training_stage_index or 0)))
    features[7] = min(float(max(0, int(getattr(battle, "turn", 0)))) / 100.0, 1.0)
    features[8] = 1.0 if bool(getattr(battle, "force_switch", False)) else 0.0

    active = getattr(battle, "active_pokemon", None)
    opp_active = getattr(battle, "opponent_active_pokemon", None)

    features[9] = 1.0 if active is not None and bool(getattr(active, "trapped", False)) else 0.0
    features[10] = min(float(len(getattr(battle, "available_moves", []) or [])) / 4.0, 1.0)
    features[11] = min(float(len(getattr(battle, "available_switches", []) or [])) / 6.0, 1.0)
    features[12] = 1.0 if bool(getattr(battle, "can_dynamax", False)) else 0.0
    features[13] = 1.0 if bool(getattr(battle, "can_mega_evolve", False)) else 0.0
    features[14] = 1.0 if bool(getattr(battle, "can_z_move", False)) else 0.0

    if active and opp_active and active.moves:
        moves = list(active.moves.values())
        for i in range(4):
            if i < len(moves):
                try:
                    mult = opp_active.damage_multiplier(moves[i])
                    features[15 + i] = min(float(mult) / 4.0, 1.0)
                except Exception:
                    features[15 + i] = 0.25

    if active and opp_active and getattr(opp_active, "moves", None):
        opp_moves = list(opp_active.moves.values())
        for i in range(4):
            if i < len(opp_moves):
                try:
                    mult = type_multiplier(opp_moves[i], active)
                    features[19 + i] = min(float(mult) / 4.0, 1.0)
                except Exception:
                    features[19 + i] = 0.25

    if active is not None and opp_active is not None:
        features[23] = 1.0 if speed_order(active, opp_active) > 0 else 0.0
        dmg = best_damage_frac(active, opp_active)
        features[26] = 1.0 if dmg >= 1.0 else 0.0
        features[27] = 1.0 if dmg >= 0.5 else 0.0

    features[24] = min(max(float(last_damage_dealt), 0.0), 1.0)
    features[25] = min(max(float(last_damage_taken), 0.0), 1.0)

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
    """Generate action mask for valid actions using poke-env natively."""
    # pylint: disable=no-member
    native_mask = SinglesEnv.get_action_mask(battle)
    mask = np.array(native_mask, dtype=np.float32)
    
    # Ensure it matches our tensor shapes
    if len(mask) >= ACTION_SPACE_N:
        return mask[:ACTION_SPACE_N]
    
    padded = np.zeros(ACTION_SPACE_N, dtype=np.float32)
    padded[:len(mask)] = mask
    return padded

def get_valid_action_indices(battle: AbstractBattle) -> List[int]:
    """Get list of valid action indices."""
    return [i for i, valid in enumerate(get_action_mask(battle)) if valid > 0.5]


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
