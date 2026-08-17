from poke_env.battle.pokemon_type import PokemonType

from src.models.matchup import (
    best_offensive_multiplier,
    speed_order,
    tank_vs_active,
    type_multiplier,
)
from tests.fakes import (
    FakeMon,
    electric_only_jolteon,
    garchomp_like,
    ice_attacker_like,
    jolteon_like,
    make_move,
)


def test_ground_beats_electric():
    eq = make_move("earthquake")
    jolteon = jolteon_like()
    assert type_multiplier(eq, jolteon) == 2.0


def test_electric_does_not_hit_ground():
    bolt = make_move("thunderbolt")
    garchomp = garchomp_like()
    assert type_multiplier(bolt, garchomp) == 0.0


def test_garchomp_outspeeds_blissey_like_bulk():
    chomp = garchomp_like()
    slow = FakeMon(
        "slowbro",
        types=(PokemonType.WATER, PokemonType.PSYCHIC),
        base_stats={"hp": 95, "atk": 75, "def": 110, "spa": 100, "spd": 80, "spe": 30},
        moves={"scald": make_move("scald")},
    )
    assert speed_order(chomp, slow) == 1.0
    assert speed_order(slow, chomp) == -1.0


def test_tank_vs_active_depends_on_opponent_typing():
    bulky_ground = garchomp_like()
    bulky_ground.moves = {"earthquake": make_move("earthquake")}
    vs_electric = tank_vs_active(bulky_ground, electric_only_jolteon())
    vs_ice = tank_vs_active(bulky_ground, ice_attacker_like())
    assert vs_electric > vs_ice
    assert vs_ice == 0.0


def test_best_offensive_multiplier_uses_known_moves_only():
    attacker = jolteon_like()
    attacker.moves = {"thunderbolt": make_move("thunderbolt")}
    water = FakeMon(
        "vaporeon",
        types=(PokemonType.WATER,),
        base_stats={"hp": 130, "atk": 65, "def": 60, "spa": 110, "spd": 95, "spe": 65},
    )
    assert best_offensive_multiplier(attacker, water) == 2.0
