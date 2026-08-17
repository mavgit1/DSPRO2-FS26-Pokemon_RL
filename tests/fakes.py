"""Shared fakes for observation / role / matchup tests."""

from __future__ import annotations

from typing import Any, Dict, Optional

from poke_env.battle.move import Move
from poke_env.battle.pokemon_type import PokemonType
from poke_env.data import GenData


def make_move(move_id: str, gen: int = 8) -> Move:
    return Move(move_id, gen=gen)


class FakeMon:
    """Duck-typed Pokemon for unit tests (poke-env Pokemon.level has no setter)."""

    def __init__(
        self,
        species: str = "blissey",
        *,
        level: int = 85,
        types: tuple = (PokemonType.NORMAL,),
        base_stats: Optional[Dict[str, int]] = None,
        stats: Optional[Dict[str, Optional[float]]] = None,
        moves: Optional[Dict[str, Any]] = None,
        item: Any = None,
        ability: Any = None,
        hp_frac: float = 1.0,
        fainted: bool = False,
        boosts: Optional[Dict[str, int]] = None,
        is_opponent: bool = False,
    ):
        self.species = species
        self.name = species
        self.level = level
        self.type_1 = types[0] if types else None
        self.type_2 = types[1] if len(types) > 1 else None
        self.types = types
        self.base_stats = base_stats or {
            "hp": 100,
            "atk": 100,
            "def": 100,
            "spa": 100,
            "spd": 100,
            "spe": 100,
        }
        self.stats = stats or {k: None for k in self.base_stats}
        self.moves = moves or {}
        self.item = item
        self.ability = ability
        self.current_hp_fraction = hp_frac
        self.fainted = fainted
        self.boosts = boosts or {}
        self.effects = {}
        self.status = None
        self.weight = 40.0
        self.trapped = False
        self.base_species = species
        self.available_z_moves = []
        self.gen = 8

    def damage_multiplier(self, move: Any) -> float:
        move_type = getattr(move, "type", None)
        if move_type is None:
            return 1.0
        types = tuple(t for t in self.types if t is not None)
        if not types:
            return 1.0
        return float(
            move_type.damage_multiplier(
                *types, type_chart=GenData.from_gen(self.gen).type_chart
            )
        )


class FakeBattle:
    def __init__(
        self,
        our_active: Any,
        opp_active: Any,
        *,
        our_bench=None,
        opp_bench=None,
        turn: int = 5,
    ):
        self.active_pokemon = our_active
        self.opponent_active_pokemon = opp_active
        team = {}
        if our_active is not None:
            team["p1a"] = our_active
        for i, mon in enumerate(our_bench or []):
            team[f"p1b{i}"] = mon
        opp_team = {}
        if opp_active is not None:
            opp_team["p2a"] = opp_active
        for i, mon in enumerate(opp_bench or []):
            opp_team[f"p2b{i}"] = mon
        self.team = team
        self.opponent_team = opp_team
        self.weather = {}
        self.fields = {}
        self.side_conditions = {}
        self.opponent_side_conditions = {}
        self.turn = turn
        self.force_switch = False
        self.trapped = False
        self._wait = False
        self.available_moves = list((our_active.moves or {}).values()) if our_active else []
        self.available_switches = []
        self.can_dynamax = False
        self.can_mega_evolve = False
        self.can_z_move = False
        self.can_tera = False
        self.won = False
        self.lost = False
        self.battle_tag = "test-battle"
        self.gen = 8


def blissey_like() -> FakeMon:
    return FakeMon(
        "blissey",
        level=85,
        types=(PokemonType.NORMAL,),
        base_stats={"hp": 255, "atk": 10, "def": 10, "spa": 75, "spd": 135, "spe": 55},
        moves={
            "softboiled": make_move("softboiled"),
            "toxic": make_move("toxic"),
            "protect": make_move("protect"),
            "seismictoss": make_move("seismictoss"),
        },
        item="leftovers",
    )


def garchomp_like() -> FakeMon:
    return FakeMon(
        "garchomp",
        level=80,
        types=(PokemonType.DRAGON, PokemonType.GROUND),
        base_stats={"hp": 108, "atk": 130, "def": 95, "spa": 80, "spd": 85, "spe": 102},
        moves={
            "swordsdance": make_move("swordsdance"),
            "earthquake": make_move("earthquake"),
            "stoneedge": make_move("stoneedge"),
            "firefang": make_move("firefang"),
        },
        item="lifeorb",
    )


def electric_only_jolteon() -> FakeMon:
    """Jolteon with only Thunderbolt so Ground immunities actually show up."""
    mon = jolteon_like()
    mon.moves = {"thunderbolt": make_move("thunderbolt")}
    return mon


def ice_attacker_like() -> FakeMon:
    """Special Ice attacker; Ice is 4x vs Dragon/Ground."""
    return FakeMon(
        "weavile",
        level=82,
        types=(PokemonType.DARK, PokemonType.ICE),
        base_stats={"hp": 70, "atk": 120, "def": 65, "spa": 45, "spd": 85, "spe": 125},
        moves={"icebeam": make_move("icebeam")},
    )


def jolteon_like() -> FakeMon:
    return FakeMon(
        "jolteon",
        level=82,
        types=(PokemonType.ELECTRIC,),
        base_stats={"hp": 65, "atk": 65, "def": 60, "spa": 110, "spd": 95, "spe": 130},
        moves={
            "thunderbolt": make_move("thunderbolt"),
            "voltswitch": make_move("voltswitch"),
            "shadowball": make_move("shadowball"),
            "toxic": make_move("toxic"),
        },
        item="choicespecs",
    )
