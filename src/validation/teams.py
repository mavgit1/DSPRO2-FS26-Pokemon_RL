from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from poke_env.teambuilder.constant_teambuilder import ConstantTeambuilder


def load_team_manifest(path: str | Path) -> Dict[str, Any]:
    """Load a generated validation team manifest."""
    manifest_path = Path(path).expanduser().resolve()
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def validate_fixed_pair_manifest(manifest: Dict[str, Any]) -> None:
    """Validate the planned Tier 2 fixed-paired team manifest shape."""
    teams = manifest.get("teams")
    pairs = manifest.get("pairs")
    if not isinstance(teams, list) or not isinstance(pairs, list):
        raise ValueError("Team manifest must contain list fields: teams and pairs.")
    if len(teams) != 20:
        raise ValueError(f"Expected 20 validation teams, got {len(teams)}.")
    if len(pairs) != 10:
        raise ValueError(f"Expected 10 validation pairs, got {len(pairs)}.")

    team_by_id = {}
    for team in teams:
        team_id = team.get("id")
        showdown = team.get("showdown")
        structured = team.get("pokemon")
        if not isinstance(team_id, str) or not team_id:
            raise ValueError("Each validation team needs a non-empty string id.")
        if team_id in team_by_id:
            raise ValueError(f"Duplicate validation team id: {team_id}")
        if not isinstance(showdown, str) or not showdown.strip():
            raise ValueError(f"Team {team_id} needs non-empty Showdown text.")
        if not isinstance(structured, list) or len(structured) != 6:
            raise ValueError(f"Team {team_id} needs structured pokemon metadata.")
        team_by_id[team_id] = team

    seen_pair_ids = set()
    for pair in pairs:
        pair_id = pair.get("id")
        team_a = pair.get("team_a")
        team_b = pair.get("team_b")
        if not isinstance(pair_id, str) or not pair_id:
            raise ValueError("Each validation pair needs a non-empty string id.")
        if pair_id in seen_pair_ids:
            raise ValueError(f"Duplicate validation pair id: {pair_id}")
        seen_pair_ids.add(pair_id)
        if team_a not in team_by_id or team_b not in team_by_id:
            raise ValueError(f"Pair {pair_id} references unknown team ids.")
        if team_a == team_b:
            raise ValueError(f"Pair {pair_id} must reference two different teams.")


def validate_mirror_manifest(manifest: Dict[str, Any]) -> None:
    """Validate the planned Tier 3 mirror team manifest shape."""
    teams = manifest.get("teams")
    if not isinstance(teams, list):
        raise ValueError("Mirror team manifest must contain a list field: teams.")
    if len(teams) != 20:
        raise ValueError(f"Expected 20 mirror teams, got {len(teams)}.")

    seen_team_ids = set()
    for team in teams:
        team_id = team.get("id")
        showdown = team.get("showdown")
        structured = team.get("pokemon")
        if not isinstance(team_id, str) or not team_id:
            raise ValueError("Each mirror team needs a non-empty string id.")
        if team_id in seen_team_ids:
            raise ValueError(f"Duplicate mirror team id: {team_id}")
        seen_team_ids.add(team_id)
        if not isinstance(showdown, str) or not showdown.strip():
            raise ValueError(f"Mirror team {team_id} needs non-empty Showdown text.")
        if not isinstance(structured, list) or len(structured) != 6:
            raise ValueError(f"Mirror team {team_id} needs structured pokemon metadata.")


def team_lookup(manifest: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Return validation teams keyed by team id."""
    return {team["id"]: team for team in manifest.get("teams", [])}


def fixed_pair_battle_specs(manifest: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Expand a fixed-paired manifest into the 40 Tier 2 battle specs."""
    validate_fixed_pair_manifest(manifest)
    teams = team_lookup(manifest)
    specs: List[Dict[str, Any]] = []

    for pair in manifest["pairs"]:
        pair_id = pair["id"]
        team_a = teams[pair["team_a"]]
        team_b = teams[pair["team_b"]]
        for opponent_type in ("random", "heuristic"):
            specs.append(
                {
                    "pair_id": pair_id,
                    "opponent_type": opponent_type,
                    "rl_team_id": team_a["id"],
                    "opponent_team_id": team_b["id"],
                    "rl_team": team_a["showdown"],
                    "opponent_team": team_b["showdown"],
                }
            )
            specs.append(
                {
                    "pair_id": pair_id,
                    "opponent_type": opponent_type,
                    "rl_team_id": team_b["id"],
                    "opponent_team_id": team_a["id"],
                    "rl_team": team_b["showdown"],
                    "opponent_team": team_a["showdown"],
                }
            )

    return specs


def mirror_battle_specs(manifest: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Expand a mirror manifest into the 40 Tier 3 battle specs."""
    validate_mirror_manifest(manifest)
    specs: List[Dict[str, Any]] = []

    for team in manifest["teams"]:
        for opponent_type in ("random", "heuristic"):
            specs.append(
                {
                    "pair_id": f"mirror_{team['id']}",
                    "opponent_type": opponent_type,
                    "rl_team_id": team["id"],
                    "opponent_team_id": team["id"],
                    "rl_team": team["showdown"],
                    "opponent_team": team["showdown"],
                }
            )

    return specs


def structured_team_from_showdown(showdown: str) -> List[Dict[str, Any]]:
    """Parse Showdown team text into inspectable structured metadata."""
    builder = ConstantTeambuilder(showdown)
    structured = []
    for mon in builder.team:
        structured.append(
            {
                "nickname": mon.nickname,
                "species": mon.species,
                "item": mon.item,
                "ability": mon.ability,
                "moves": mon.moves or [],
                "nature": mon.nature,
                "evs": mon.evs,
                "ivs": mon.ivs,
                "gender": mon.gender,
                "level": mon.level,
                "shiny": mon.shiny,
            }
        )
    return structured
