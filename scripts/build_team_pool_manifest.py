#!/usr/bin/env python3
"""Build gen8_pool_{3,5,10,20}.json from the full validation team manifest."""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE = REPO_ROOT / "data/validation/gen8_random_battle_team_pairs.json"
OUT_DIR = REPO_ROOT / "data/teams/pool_curriculum"
SIZES = (3, 5, 10, 20)


def main() -> int:
    full = json.loads(SOURCE.read_text(encoding="utf-8"))
    teams = full["teams"]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for n in SIZES:
        if n > len(teams):
            print(f"error: only {len(teams)} teams in source, cannot build pool {n}", file=sys.stderr)
            return 1
        meta = dict(full.get("metadata") or {})
        meta["team_count"] = n
        meta["pool_curriculum"] = "first_n_teams_from_validation_set"
        meta["source_manifest"] = str(SOURCE.relative_to(REPO_ROOT))
        out = {"metadata": meta, "teams": teams[:n]}
        path = OUT_DIR / f"gen8_pool_{n}.json"
        path.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {path} ({n} teams)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
