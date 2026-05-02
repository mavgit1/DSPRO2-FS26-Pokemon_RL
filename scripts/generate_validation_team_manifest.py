#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.validation.teams import validate_fixed_pair_manifest  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate fixed paired validation teams from Pokemon Showdown."
    )
    parser.add_argument(
        "--format",
        default="gen8randombattle",
        help="Pokemon Showdown format to generate teams from.",
    )
    parser.add_argument(
        "--execution-format",
        default="gen8customgame",
        help="Pokemon Showdown format used to replay the generated fixed teams.",
    )
    parser.add_argument(
        "--team-count",
        type=int,
        default=20,
        help="Number of teams to generate. Must be even.",
    )
    parser.add_argument(
        "--seed",
        default="pokemon-rl-validation-v1",
        help="Seed prefix used for deterministic Showdown team generation.",
    )
    parser.add_argument(
        "--showdown-dir",
        type=Path,
        default=Path("pokemon-showdown"),
        help="Path to the local Pokemon Showdown checkout.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/validation/gen8_random_battle_team_pairs.json"),
        help="Output manifest path.",
    )
    args = parser.parse_args()

    if args.team_count <= 0 or args.team_count % 2 != 0:
        raise SystemExit("--team-count must be a positive even number.")

    generated = _generate_showdown_teams(
        showdown_dir=args.showdown_dir,
        battle_format=args.format,
        team_count=args.team_count,
        seed=args.seed,
    )
    manifest = _build_manifest(
        generated=generated,
        battle_format=args.format,
        execution_format=args.execution_format,
        seed=args.seed,
        showdown_dir=args.showdown_dir,
    )
    validate_fixed_pair_manifest(manifest)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"Wrote {args.team_count} validation teams to {args.output}")
    print(f"Pairs: {len(manifest['pairs'])}")
    return 0


def _generate_showdown_teams(
    showdown_dir: Path,
    battle_format: str,
    team_count: int,
    seed: str,
) -> List[Dict[str, Any]]:
    showdown_root = showdown_dir.expanduser().resolve()
    node_script = """
const path = require('path');
const crypto = require('crypto');
const showdownRoot = process.argv[1];
const battleFormat = process.argv[2];
const teamCount = Number(process.argv[3]);
const seedPrefix = process.argv[4];
const {Teams} = require(path.join(showdownRoot, 'dist', 'sim'));

function seedArray(seedText) {
  const digest = crypto.createHash('sha256').update(seedText).digest();
  return [
    digest.readUInt32BE(0),
    digest.readUInt32BE(4),
    digest.readUInt32BE(8),
    digest.readUInt32BE(12),
  ];
}

const out = [];
for (let i = 0; i < teamCount; i++) {
  const seed = seedArray(`${seedPrefix}-${String(i + 1).padStart(2, '0')}`);
  const pokemon = Teams.generate(battleFormat, {seed});
  out.push({
    id: `team_${String(i + 1).padStart(2, '0')}`,
    seed,
    showdown: Teams.export(pokemon),
    pokemon,
  });
}
process.stdout.write(JSON.stringify(out));
"""
    proc = subprocess.run(
        [
            "node",
            "-e",
            node_script,
            str(showdown_root),
            battle_format,
            str(team_count),
            seed,
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(proc.stdout)


def _build_manifest(
    generated: List[Dict[str, Any]],
    battle_format: str,
    execution_format: str,
    seed: str,
    showdown_dir: Path,
) -> Dict[str, Any]:
    teams = [
        {
            "id": item["id"],
            "seed": item["seed"],
            "showdown": item["showdown"],
            "pokemon": item["pokemon"],
        }
        for item in generated
    ]
    pairs = []
    for idx in range(0, len(teams), 2):
        pair_number = (idx // 2) + 1
        pairs.append(
            {
                "id": f"pair_{pair_number:02d}",
                "team_a": teams[idx]["id"],
                "team_b": teams[idx + 1]["id"],
            }
        )

    return {
        "metadata": {
            "generation_format": battle_format,
            "execution_format": execution_format,
            "team_count": len(teams),
            "pair_count": len(pairs),
            "seed": seed,
            "generated_at": datetime.now(UTC).isoformat(),
            "generator": "pokemon-showdown Teams.generate",
            "showdown_dir": str(showdown_dir),
        },
        "teams": teams,
        "pairs": pairs,
    }


if __name__ == "__main__":
    raise SystemExit(main())
