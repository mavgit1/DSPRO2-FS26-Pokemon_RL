#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.vocab import get_embedding_vocab, normalize_dex_id  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Audit species/item/ability coverage for embedding vocabularies."
    )
    parser.add_argument(
        "--bdsp-dataset",
        type=Path,
        default=Path("data/bdsp_trainers.json"),
        help="BDSP trainer dataset JSON to audit.",
    )
    parser.add_argument(
        "--validation-manifest",
        type=Path,
        default=Path("data/validation/gen8_random_battle_team_pairs.json"),
        help="Validation team manifest JSON to audit.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("logs/validation/embedding_vocab_audit.json"),
        help="Audit report output path.",
    )
    args = parser.parse_args()

    vocab = get_embedding_vocab()
    report = {
        "vocab_sizes": {
            "species": vocab.species_vocab_size,
            "items": vocab.item_vocab_size,
            "abilities": vocab.ability_vocab_size,
        },
        "sources": {},
    }

    if args.bdsp_dataset.exists():
        report["sources"]["bdsp_dataset"] = audit_pokemon_records(
            _iter_bdsp_pokemon(args.bdsp_dataset),
            source=str(args.bdsp_dataset),
        )

    if args.validation_manifest.exists():
        report["sources"]["validation_manifest"] = audit_pokemon_records(
            _iter_manifest_pokemon(args.validation_manifest),
            source=str(args.validation_manifest),
        )

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"Wrote vocab audit to {args.output_json}")
    for source_name, source_report in report["sources"].items():
        print(
            f"{source_name}: "
            f"species_unknown={source_report['unknown_counts']['species']} "
            f"items_unknown={source_report['unknown_counts']['items']} "
            f"abilities_unknown={source_report['unknown_counts']['abilities']}"
        )

    return 0


def audit_pokemon_records(
    records: Iterable[dict[str, Any]],
    source: str,
) -> dict[str, Any]:
    vocab = get_embedding_vocab()
    unknown_species: Counter[str] = Counter()
    unknown_items: Counter[str] = Counter()
    unknown_abilities: Counter[str] = Counter()
    total = 0

    for mon in records:
        total += 1
        species = mon.get("species") or mon.get("name")
        item = mon.get("item")
        ability = mon.get("ability")

        if species and vocab.species_id(species) == 0:
            unknown_species[str(species)] += 1
        if item and vocab.item_id(item) == 0:
            unknown_items[str(item)] += 1
        if ability and vocab.ability_id(ability) == 0:
            unknown_abilities[str(ability)] += 1

    return {
        "source": source,
        "pokemon_count": total,
        "unknown_counts": {
            "species": int(sum(unknown_species.values())),
            "items": int(sum(unknown_items.values())),
            "abilities": int(sum(unknown_abilities.values())),
        },
        "unknown_values": {
            "species": _counter_payload(unknown_species),
            "items": _counter_payload(unknown_items),
            "abilities": _counter_payload(unknown_abilities),
        },
    }


def _iter_bdsp_pokemon(path: Path) -> Iterable[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    for trainer in payload.get("trainers", []):
        yield from trainer.get("party", [])


def _iter_manifest_pokemon(path: Path) -> Iterable[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    for team in payload.get("teams", []):
        yield from team.get("pokemon", [])


def _counter_payload(counter: Counter[str]) -> list[dict[str, Any]]:
    return [
        {
            "value": value,
            "normalized": normalize_dex_id(value),
            "count": count,
        }
        for value, count in counter.most_common()
    ]


if __name__ == "__main__":
    raise SystemExit(main())
