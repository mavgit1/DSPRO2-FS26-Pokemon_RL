import numpy as np

from src.models.embedding import (
    GLOBAL_EXTRA_FEATURE_NAMES,
    GLOBAL_EXTRA_START_IDX,
    ITEM_CLASS_START,
    KINEMATIC_START,
    MATCHUP_START,
    NUM_TOKENS,
    ROLE_START,
    TOKEN_DIM,
    embed_battle,
    embed_pokemon,
)
from src.models.role_scores import ROLE_SCORE_NAMES
from tests.fakes import (
    FakeBattle,
    blissey_like,
    electric_only_jolteon,
    garchomp_like,
    ice_attacker_like,
    jolteon_like,
)


def test_token_dim_is_legacy_plus_tactical_block():
    from src.models.embedding import (
        ITEM_CLASS_DIM,
        KINEMATIC_DIM,
        LEGACY_POKEMON_DIM,
        MATCHUP_DIM,
        ROLE_DIM,
    )

    assert TOKEN_DIM == LEGACY_POKEMON_DIM + KINEMATIC_DIM + ROLE_DIM + ITEM_CLASS_DIM + MATCHUP_DIM
    assert MATCHUP_START + MATCHUP_DIM == TOKEN_DIM
    obs = embed_pokemon(blissey_like())
    assert obs["obs"].shape == (TOKEN_DIM,)
    assert obs["obs"].dtype == np.float32
    assert not np.isnan(obs["obs"]).any()


def test_empty_slot_is_flagged_and_otherwise_zero():
    obs = embed_pokemon(None)["obs"]
    assert obs[0] == 1.0
    assert obs[1:].sum() == 0.0


def test_blissey_level_and_leftovers_and_roles_are_written():
    obs = embed_pokemon(blissey_like(), is_active=True)["obs"]
    assert obs[KINEMATIC_START] == np.float32(0.85)
    leftovers_idx = ITEM_CLASS_START + 1
    assert obs[leftovers_idx] == 1.0
    stall_idx = ROLE_START + ROLE_SCORE_NAMES.index("stall")
    sweeper_idx = ROLE_START + ROLE_SCORE_NAMES.index("sweeper")
    assert obs[stall_idx] > obs[sweeper_idx]


def test_earthquake_stab_and_closecombat_contact_on_garchomp():
    from tests.fakes import make_move

    chomp = garchomp_like()
    chomp.moves = {
        "swordsdance": make_move("swordsdance"),
        "earthquake": make_move("earthquake"),
        "closecombat": make_move("closecombat"),
        "stoneedge": make_move("stoneedge"),
    }
    obs = embed_pokemon(chomp)["obs"]
    eq_slot = KINEMATIC_START + 7 + 4
    assert obs[eq_slot + 2] == 1.0  # STAB
    cc_slot = KINEMATIC_START + 7 + 8
    assert obs[cc_slot + 3] == 1.0  # contact


def test_embed_battle_shape_and_last_damage():
    battle = FakeBattle(garchomp_like(), jolteon_like())
    packed = embed_battle(battle, last_damage_dealt=0.4, last_damage_taken=0.1)
    assert packed["obs"].shape == (NUM_TOKENS, TOKEN_DIM)
    assert packed["action_mask"].shape[0] >= 14
    dealt_idx = GLOBAL_EXTRA_START_IDX + GLOBAL_EXTRA_FEATURE_NAMES.index(
        "last_damage_dealt_frac"
    )
    taken_idx = GLOBAL_EXTRA_START_IDX + GLOBAL_EXTRA_FEATURE_NAMES.index(
        "last_damage_taken_frac"
    )
    assert packed["obs"][0, dealt_idx] == np.float32(0.4)
    assert packed["obs"][0, taken_idx] == np.float32(0.1)


def test_tank_feature_higher_into_resisted_attacker():
    chomp = garchomp_like()
    vs_electric = embed_pokemon(chomp, vs_mon=electric_only_jolteon())["obs"][
        MATCHUP_START + 4
    ]
    vs_ice = embed_pokemon(chomp, vs_mon=ice_attacker_like())["obs"][MATCHUP_START + 4]
    assert vs_electric > vs_ice
    assert vs_ice == 0.0


def test_opponent_hidden_species_does_not_get_roles():
    hidden = jolteon_like()
    hidden.species = "unknown"
    hidden.moves = {}
    obs = embed_pokemon(hidden, is_opponent=True)["obs"]
    assert obs[ROLE_START : MATCHUP_START].sum() == 0.0
