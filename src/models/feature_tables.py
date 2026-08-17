"""Move-tag and item-class tables derived from Showdown / poke-env dex data.

Role *labels* from random-battle ``sets.json`` are not used here. Tags are
computed from the move itself so they work on gauntlet teams.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set

from src.models.vocab import normalize_dex_id

FEATURE_DIR = Path(__file__).resolve().parents[2] / "data/features"
MOVE_TAGS_PATH = FEATURE_DIR / "move_tags.json"

TAG_NAMES: List[str] = [
    "recovery",
    "setup",
    "pivot",
    "hazard",
    "hazard_control",
    "priority",
    "status",
    "drain",
    "recoil",
    "screens",
    "speed_control",
    "protect",
    "contact",
]

ITEM_CLASS_NAMES: List[str] = [
    "choice",
    "leftovers",
    "sash",
    "boots",
    "lifeorb",
    "vest",
]

_ITEM_CLASS_IDS: Dict[str, str] = {
    "choiceband": "choice",
    "choicespecs": "choice",
    "choicescarf": "choice",
    "leftovers": "leftovers",
    "blacksludge": "leftovers",
    "focussash": "sash",
    "heavydutyboots": "boots",
    "lifeorb": "lifeorb",
    "assaultvest": "vest",
}

# Moves whose function is not obvious from poke-env fields alone.
_EXTRA_TAGS: Dict[str, List[str]] = {
    "rapidspin": ["hazard_control"],
    "defog": ["hazard_control"],
    "courtchange": ["hazard_control"],
    "mortalspin": ["hazard_control"],
    "tidyup": ["hazard_control"],
    "wish": ["recovery"],
    "painsplit": ["recovery"],
    "strengthsap": ["recovery"],
    "leechseed": ["status"],
    "toxic": ["status"],
    "willowisp": ["status"],
    "thunderwave": ["status", "speed_control"],
    "glare": ["status", "speed_control"],
    "stunspore": ["status", "speed_control"],
    "nuzzle": ["speed_control"],
    "icywind": ["speed_control"],
    "electroweb": ["speed_control"],
    "stringshot": ["speed_control"],
    "bulldoze": ["speed_control"],
    "rocktomb": ["speed_control"],
    "lightscreen": ["screens"],
    "reflect": ["screens"],
    "auroraveil": ["screens"],
    "tailwind": ["speed_control"],
    "trickroom": ["speed_control"],
    "teleport": ["pivot"],
    "partingshot": ["pivot"],
    "voltswitch": ["pivot"],
    "uturn": ["pivot"],
    "flipturn": ["pivot"],
    "batonpass": ["pivot"],
    "chillyreception": ["pivot"],
    "shedtail": ["pivot"],
}


def _move_key(move: Any) -> str:
    if move is None:
        return ""
    if isinstance(move, str):
        return normalize_dex_id(move)
    move_id = getattr(move, "id", None) or getattr(move, "name", None)
    return normalize_dex_id(move_id)


def _category_name(move: Any) -> str:
    category = getattr(move, "category", None)
    if category is None and isinstance(move, Mapping):
        category = move.get("category")
    name = getattr(category, "name", None) or str(category or "")
    return name.upper()


def infer_tags_from_move(move: Any) -> Set[str]:
    """Infer tactical tags from a poke-env Move, GenData dict, or move id."""
    tags: Set[str] = set()
    key = _move_key(move)
    if not key:
        return tags

    tags.update(_EXTRA_TAGS.get(key, ()))

    if isinstance(move, str):
        return tags

    data: Mapping[str, Any]
    if isinstance(move, Mapping):
        data = move
    else:
        data = {}

    category = _category_name(move) if not data else str(data.get("category", "")).upper()
    if category == "STATUS":
        tags.add("status")

    priority = getattr(move, "priority", None)
    if priority is None:
        priority = data.get("priority", 0)
    try:
        if float(priority) > 0 and category != "STATUS":
            tags.add("priority")
    except (TypeError, ValueError):
        pass

    heal = getattr(move, "heal", None)
    if heal is None:
        heal = data.get("heal", 0)
    try:
        if float(heal or 0) > 0:
            tags.add("recovery")
    except (TypeError, ValueError):
        pass

    drain = getattr(move, "drain", None)
    if drain is None:
        drain = data.get("drain", 0)
    try:
        if float(drain or 0) > 0:
            tags.add("drain")
    except (TypeError, ValueError):
        pass

    recoil = getattr(move, "recoil", None)
    if recoil is None:
        recoil = data.get("recoil", 0)
    if recoil:
        tags.add("recoil")

    if getattr(move, "self_switch", None) or data.get("selfSwitch"):
        tags.add("pivot")

    if getattr(move, "is_protect_move", False) or data.get("stallingMove"):
        tags.add("protect")
    flags = getattr(move, "flags", None) or data.get("flags") or ()
    flag_names = {str(f).lower() for f in flags} if not isinstance(flags, Mapping) else {
        str(k).lower() for k, v in flags.items() if v
    }
    if "contact" in flag_names:
        tags.add("contact")
    if "protect" in flag_names and getattr(move, "stalling_move", False):
        tags.add("protect")

    boosts = getattr(move, "boosts", None) or getattr(move, "self_boost", None) or data.get("boosts")
    if boosts and category == "STATUS":
        tags.add("setup")

    side = getattr(move, "side_condition", None)
    side_name = getattr(side, "name", None) or str(side or data.get("sideCondition") or "")
    side_key = normalize_dex_id(side_name)
    if side_key in {"stealthrock", "spikes", "toxicspikes", "stickyweb"}:
        tags.add("hazard")

    return tags


@lru_cache(maxsize=1)
def load_move_tags() -> Dict[str, List[str]]:
    if not MOVE_TAGS_PATH.is_file():
        return {key: list(tags) for key, tags in _EXTRA_TAGS.items()}
    payload = json.loads(MOVE_TAGS_PATH.read_text(encoding="utf-8"))
    raw = payload.get("moves", payload)
    return {normalize_dex_id(k): list(v) for k, v in raw.items()}


def tags_for_move(move: Any) -> Set[str]:
    key = _move_key(move)
    tags = set(infer_tags_from_move(move))
    table = load_move_tags()
    tags.update(table.get(key, ()))
    tags.update(_EXTRA_TAGS.get(key, ()))
    return {t for t in tags if t in TAG_NAMES}


def tags_for_moves(moves: Optional[Iterable[Any]]) -> Set[str]:
    out: Set[str] = set()
    if not moves:
        return out
    values: Sequence[Any]
    if isinstance(moves, Mapping):
        values = list(moves.values())
    else:
        values = list(moves)
    for move in values:
        out.update(tags_for_move(move))
    return out


def item_class_vector(item: Any) -> List[float]:
    flags = [0.0] * len(ITEM_CLASS_NAMES)
    key = normalize_dex_id(item)
    if not key or key in {"unknown", "unknownitem", ""}:
        return flags
    cls = _ITEM_CLASS_IDS.get(key)
    if cls is None:
        return flags
    flags[ITEM_CLASS_NAMES.index(cls)] = 1.0
    return flags
