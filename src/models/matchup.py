"""Cheap type/damage/speed sheet. No smogon/calc — table lookups and arithmetic."""

from __future__ import annotations

from typing import Any, Mapping, Sequence, Tuple

import numpy as np
from poke_env.data import GenData

from src.models.role_scores import estimate_stats

_TYPE_CHART = GenData.from_gen(8).type_chart


def _types_of(mon: Any) -> Tuple[Any, ...]:
    if mon is None:
        return ()
    types = getattr(mon, "types", None)
    if types:
        return tuple(t for t in types if t is not None)
    out = []
    for attr in ("type_1", "type_2"):
        t = getattr(mon, attr, None)
        if t is not None:
            out.append(t)
    return tuple(out)


def _moves_of(mon: Any) -> Sequence[Any]:
    moves = getattr(mon, "moves", None)
    if not moves:
        return ()
    if isinstance(moves, Mapping):
        return [m for m in moves.values() if m is not None]
    return [m for m in moves if m is not None]


def _is_damaging(move: Any) -> bool:
    category = getattr(move, "category", None)
    name = getattr(category, "name", None) or str(category or "")
    return name.upper() != "STATUS"


def type_multiplier(move: Any, target: Any) -> float:
    if move is None or target is None:
        return 1.0
    types = _types_of(target)
    if not types:
        return 1.0
    move_type = getattr(move, "type", None)
    if move_type is None:
        return 1.0
    try:
        return float(
            move_type.damage_multiplier(*types, type_chart=_TYPE_CHART)
        )
    except Exception:
        try:
            if hasattr(target, "damage_multiplier"):
                return float(target.damage_multiplier(move))
        except Exception:
            return 1.0
    return 1.0


def best_offensive_multiplier(attacker: Any, defender: Any) -> float:
    best = 0.0
    found = False
    for move in _moves_of(attacker):
        if not _is_damaging(move):
            continue
        best = max(best, type_multiplier(move, defender))
        found = True
    return best if found else 0.0


def boosted_spe(mon: Any) -> float:
    if mon is None:
        return 0.0
    stats = estimate_stats(mon)
    return float(stats.get("spe", 0.0))


def speed_order(left: Any, right: Any) -> float:
    """-1 if left is slower, 0 if tie, +1 if left is faster."""
    if left is None or right is None:
        return 0.0
    diff = boosted_spe(left) - boosted_spe(right)
    if abs(diff) < 1e-6:
        return 0.0
    return 1.0 if diff > 0 else -1.0


def spe_ratio(left: Any, right: Any) -> float:
    if left is None or right is None:
        return 0.0
    a = boosted_spe(left)
    b = boosted_spe(right)
    denom = max(a, b, 1.0)
    return float(np.clip((a - b) / denom, -1.0, 1.0))


def simplified_damage_frac(attacker: Any, defender: Any, move: Any) -> float:
    """Very rough HP-fraction damage. Clipped to [0, 2]."""
    if attacker is None or defender is None or move is None or not _is_damaging(move):
        return 0.0
    bp = float(getattr(move, "base_power", 0) or 0)
    if bp <= 0:
        return 0.0
    atk_stats = estimate_stats(attacker)
    def_stats = estimate_stats(defender)
    category = str(getattr(getattr(move, "category", None), "name", "") or "").upper()
    if category == "SPECIAL":
        off = atk_stats.get("spa", 1.0)
        deff = max(def_stats.get("spd", 1.0), 1.0)
    else:
        off = atk_stats.get("atk", 1.0)
        deff = max(def_stats.get("def", 1.0), 1.0)
    level = float(getattr(attacker, "level", 80) or 80)
    stab = 1.0
    move_type = getattr(move, "type", None)
    if move_type is not None and move_type in _types_of(attacker):
        stab = 1.5
    typ = type_multiplier(move, defender)
    # Standard damage skeleton without random, items, or crit.
    raw = (((2.0 * level / 5.0 + 2.0) * bp * off / deff) / 50.0 + 2.0) * stab * typ
    target_hp = max(def_stats.get("hp", 1.0), 1.0)
    return float(np.clip(raw / target_hp, 0.0, 2.0))


def best_damage_frac(attacker: Any, defender: Any) -> float:
    best = 0.0
    for move in _moves_of(attacker):
        best = max(best, simplified_damage_frac(attacker, defender, move))
    return best


def tank_vs_active(mon: Any, opponent: Any) -> float:
    """How well ``mon`` walls the current opponent. 0 if either side is missing."""
    if mon is None or opponent is None:
        return 0.0
    if bool(getattr(mon, "fainted", False)):
        return 0.0
    def_mult = best_offensive_multiplier(opponent, mon)
    bulk_stats = estimate_stats(mon)
    bulk = (bulk_stats.get("hp", 0.0) + bulk_stats.get("def", 0.0) + bulk_stats.get("spd", 0.0)) / 1200.0
    hp_frac = float(getattr(mon, "current_hp_fraction", 1.0) or 0.0)
    resist = 1.0 - float(np.clip(def_mult / 4.0, 0.0, 1.0))
    return float(np.clip(bulk * resist * hp_frac, 0.0, 1.0))


def matchup_vector(mon: Any, vs_mon: Any) -> np.ndarray:
    """Per-token matchup features: speed_order, spe_ratio, off, def, tank."""
    vec = np.zeros(5, dtype=np.float32)
    if mon is None or vs_mon is None:
        return vec
    vec[0] = speed_order(mon, vs_mon)
    vec[1] = spe_ratio(mon, vs_mon)
    off = best_offensive_multiplier(mon, vs_mon)
    deff = best_offensive_multiplier(vs_mon, mon)
    vec[2] = float(np.clip(off / 4.0, 0.0, 1.0))
    vec[3] = float(np.clip(deff / 4.0, 0.0, 1.0))
    vec[4] = tank_vs_active(mon, vs_mon)
    return vec
