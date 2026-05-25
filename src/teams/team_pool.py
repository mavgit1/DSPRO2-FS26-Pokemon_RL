from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Dict, List


def _load_team_manifest(path: str | Path) -> Dict[str, Any]:
    manifest_path = Path(path).expanduser().resolve()
    return json.loads(manifest_path.read_text(encoding="utf-8"))


class TeamPool:
    """Sample Showdown team text from a validation-style team manifest."""

    def __init__(self, manifest_path: str | Path):
        manifest = _load_team_manifest(manifest_path)
        teams = manifest.get("teams")
        if not isinstance(teams, list) or not teams:
            raise ValueError(f"No teams found in manifest: {manifest_path}")

        showdown_teams: List[str] = []
        for team in teams:
            showdown = team.get("showdown") if isinstance(team, dict) else None
            if isinstance(showdown, str) and showdown.strip():
                showdown_teams.append(showdown.strip())

        if not showdown_teams:
            raise ValueError(f"No Showdown teams in manifest: {manifest_path}")

        self._teams = showdown_teams

    def sample(self) -> str:
        return random.choice(self._teams)

    def __len__(self) -> int:
        return len(self._teams)
