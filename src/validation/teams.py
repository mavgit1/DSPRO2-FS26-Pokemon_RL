from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict


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
