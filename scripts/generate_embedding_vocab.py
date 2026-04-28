#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Dict, List


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate stable embedding vocabularies from Pokemon Showdown Dex."
    )
    parser.add_argument(
        "--format",
        default="gen8",
        help="Pokemon Showdown Dex mod to use for vocab generation.",
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
        default=Path("data/vocab/gen8_embedding_vocab.json"),
        help="Output vocab JSON path.",
    )
    args = parser.parse_args()

    vocab = _generate_vocab(
        showdown_dir=args.showdown_dir,
        dex_mod=args.format,
    )
    vocab["metadata"] = {
        "dex_mod": args.format,
        "generated_at": datetime.now(UTC).isoformat(),
        "generator": "pokemon-showdown Dex",
        "showdown_dir": str(args.showdown_dir),
        "padding_id": 0,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(vocab, indent=2), encoding="utf-8")

    print(f"Wrote vocab to {args.output}")
    print(f"species: {len(vocab['species'])}")
    print(f"items: {len(vocab['items'])}")
    print(f"abilities: {len(vocab['abilities'])}")
    return 0


def _generate_vocab(showdown_dir: Path, dex_mod: str) -> Dict[str, List[Dict[str, Any]]]:
    showdown_root = showdown_dir.expanduser().resolve()
    node_script = """
const path = require('path');
const showdownRoot = process.argv[1];
const dexMod = process.argv[2];
const {Dex} = require(path.join(showdownRoot, 'dist', 'sim'));

const dex = Dex.mod(dexMod);
function entries(collection) {
  return collection
    .all()
    .filter(entry => entry.exists !== false)
    .map(entry => ({id: entry.id, name: entry.name}))
    .sort((a, b) => a.id.localeCompare(b.id));
}

process.stdout.write(JSON.stringify({
  species: entries(dex.species),
  items: entries(dex.items),
  abilities: entries(dex.abilities),
}));
"""
    proc = subprocess.run(
        ["node", "-e", node_script, str(showdown_root), dex_mod],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(proc.stdout)


if __name__ == "__main__":
    raise SystemExit(main())
