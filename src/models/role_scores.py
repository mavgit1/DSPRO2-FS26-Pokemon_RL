"""Fractional role scores from revealed moves, stats, and item.

These are format-agnostic. A BDSP gauntlet Geodude with Stealth Rock scores
``hazard``/``bulky`` the same way a randbats Toxapex scores ``stall``.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional

import numpy as np

from src.models.feature_tables import tags_for_moves

BASE_STATS = ("hp", "atk", "def", "spa", "spd", "spe")

ROLE_SCORE_NAMES: List[str] = [
    "sweeper",
    "wallbreaker",
    "fast_attacker",
    "pivot",
    "support",
    "stall",
    "bulky",
]


def _clip01(value: float) -> float:
    return float(np.clip(value, 0.0, 1.0))


def estimate_stat(base: float, level: int, is_hp: bool, ev: int = 85, iv: int = 31) -> float:
    """Gen 3+ stat formula with randbats-like 85 EVs / 31 IVs / neutral nature."""
    inner = int((2 * int(base) + iv + ev // 4) * level / 100)
    if is_hp:
        return float(inner + level + 10)
    return float(inner + 5)


def estimate_stats(mon: Any) -> Dict[str, float]:
    """Return in-battle stats, falling back to the 85-EV estimate when Showdown has not sent them."""
    level = int(getattr(mon, "level", 100) or 100)
    level = max(1, min(level, 100))
    base = getattr(mon, "base_stats", None) or {}
    live = getattr(mon, "stats", None) or {}
    boosts = getattr(mon, "boosts", None) or {}
    out: Dict[str, float] = {}
    for name in BASE_STATS:
        live_val = live.get(name) if isinstance(live, Mapping) else None
        if live_val is not None:
            try:
                if float(live_val) > 0:
                    out[name] = float(live_val)
                    continue
            except (TypeError, ValueError):
                pass
        base_val = 0.0
        if isinstance(base, Mapping):
            try:
                base_val = float(base.get(name, 0) or 0)
            except (TypeError, ValueError):
                base_val = 0.0
        out[name] = estimate_stat(base_val, level, is_hp=(name == "hp"))
        if name != "hp":
            boost = 0
            if isinstance(boosts, Mapping):
                try:
                    boost = int(boosts.get(name, 0) or 0)
                except (TypeError, ValueError):
                    boost = 0
            boost = max(-6, min(6, boost))
            if boost >= 0:
                out[name] *= (2 + boost) / 2.0
            else:
                out[name] *= 2.0 / (2 - boost)
    return out


def _known_moves(mon: Any) -> List[Any]:
    moves = getattr(mon, "moves", None)
    if not moves:
        return []
    if isinstance(moves, Mapping):
        return [m for m in moves.values() if m is not None]
    return [m for m in moves if m is not None]


def _is_status_move(move: Any) -> bool:
    category = getattr(move, "category", None)
    name = getattr(category, "name", None) or str(category or "")
    return name.upper() == "STATUS"


def compute_role_scores(
    mon: Any,
    *,
    is_opponent: bool = False,
    revealed_move_frac: Optional[float] = None,
) -> np.ndarray:
    """Return a length-7 vector of role fractions in ``[0, 1]``."""
    scores = np.zeros(len(ROLE_SCORE_NAMES), dtype=np.float32)
    if mon is None:
        return scores

    known = _known_moves(mon)
    n_known = len(known)
    if revealed_move_frac is None:
        revealed_move_frac = 1.0 if not is_opponent else n_known / 4.0
    reveal = _clip01(float(revealed_move_frac))
    if reveal <= 0.0 or n_known == 0:
        return scores

    tags = tags_for_moves(known)
    has_recovery = "recovery" in tags
    has_setup = "setup" in tags
    has_pivot = "pivot" in tags
    has_protect = "protect" in tags
    status_frac = sum(1 for m in known if _is_status_move(m)) / float(n_known)

    stats = estimate_stats(mon)
    atk = stats.get("atk", 0.0)
    spa = stats.get("spa", 0.0)
    offense = _clip01(max(atk, spa) / 400.0)
    bulk = _clip01(
        (stats.get("hp", 0.0) + stats.get("def", 0.0) + stats.get("spd", 0.0)) / 1200.0
    )
    speed = _clip01(stats.get("spe", 0.0) / 400.0)

    raw = {
        "sweeper": has_setup * offense * speed,
        "wallbreaker": (1.0 - float(has_setup)) * offense * (1.0 - speed),
        "fast_attacker": (1.0 - float(has_setup)) * offense * speed,
        "pivot": float(has_pivot) * max(speed, 0.35),
        "support": float(has_recovery) * status_frac,
        "stall": float(has_recovery) * float(has_protect) * max(status_frac, 0.25),
        "bulky": bulk * (0.5 + 0.5 * float(has_recovery)),
    }
    for i, name in enumerate(ROLE_SCORE_NAMES):
        scores[i] = _clip01(raw[name] * reveal)
    return scores
