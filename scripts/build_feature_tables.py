#!/usr/bin/env python3
"""Generate data/features/move_tags.json from the poke-env Gen 8 dex."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from poke_env.battle.move import Move  # noqa: E402
from poke_env.data import GenData  # noqa: E402

from src.models.feature_tables import TAG_NAMES, infer_tags_from_move  # noqa: E402

OUT = ROOT / "data/features/move_tags.json"


def main() -> None:
    gen = GenData.from_gen(8)
    moves: dict[str, list[str]] = {}
    skipped = 0
    for move_id in gen.moves:
        try:
            move = Move(move_id, gen=8)
        except Exception:
            skipped += 1
            continue
        tags = sorted(t for t in infer_tags_from_move(move) if t in TAG_NAMES)
        if tags:
            moves[move_id] = tags
    OUT.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "gen": 8,
        "n_moves": len(moves),
        "skipped": skipped,
        "tag_names": TAG_NAMES,
        "moves": dict(sorted(moves.items())),
    }
    OUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(moves)} tagged moves to {OUT} (skipped {skipped})")


if __name__ == "__main__":
    main()
