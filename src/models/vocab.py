from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List


VOCAB_PATH = Path(__file__).resolve().parents[2] / "data/vocab/gen8_embedding_vocab.json"
PADDING_ID = 0
SPECIES_ALIASES = {
    # BDSP data uses display names that differ from Pokemon Showdown IDs.
    "shelloseastsea": "shelloseast",
    "gastrodoneastsea": "gastrodoneast",
    "washrotom": "rotomwash",
    "mowrotom": "rotommow",
    "heatrotom": "rotomheat",
    "wormadamtrashcloak": "wormadamtrash",
    "wormadamsandycloak": "wormadamsandy",
}


def normalize_dex_id(value: Any) -> str:
    """Normalize text the same way Pokemon Showdown IDs are generally formed."""
    if value is None:
        return ""
    return re.sub(r"[^a-z0-9]+", "", str(value).lower())


class EmbeddingVocab:
    """Stable lookup tables for categorical battle entities."""

    def __init__(self, payload: Dict[str, Any]):
        self.metadata = dict(payload.get("metadata", {}))
        self.species_entries = list(payload["species"])
        self.item_entries = list(payload["items"])
        self.ability_entries = list(payload["abilities"])

        self.species_to_id = self._build_lookup(self.species_entries)
        self.item_to_id = self._build_lookup(self.item_entries)
        self.ability_to_id = self._build_lookup(self.ability_entries)
        self._add_aliases(self.species_to_id, SPECIES_ALIASES)

    @staticmethod
    def _build_lookup(entries: List[Dict[str, str]]) -> Dict[str, int]:
        lookup: Dict[str, int] = {}
        for idx, entry in enumerate(entries, start=1):
            for key in (entry.get("id"), entry.get("name")):
                normalized = normalize_dex_id(key)
                if normalized:
                    lookup.setdefault(normalized, idx)
        return lookup

    @staticmethod
    def _add_aliases(lookup: Dict[str, int], aliases: Dict[str, str]) -> None:
        for alias, canonical in aliases.items():
            canonical_id = lookup.get(canonical)
            if canonical_id is not None:
                lookup.setdefault(alias, canonical_id)

    @property
    def species_vocab_size(self) -> int:
        return len(self.species_entries) + 1

    @property
    def item_vocab_size(self) -> int:
        return len(self.item_entries) + 1

    @property
    def ability_vocab_size(self) -> int:
        return len(self.ability_entries) + 1

    def species_id(self, value: Any) -> int:
        return self.species_to_id.get(normalize_dex_id(value), PADDING_ID)

    def item_id(self, value: Any) -> int:
        return self.item_to_id.get(normalize_dex_id(value), PADDING_ID)

    def ability_id(self, value: Any) -> int:
        return self.ability_to_id.get(normalize_dex_id(value), PADDING_ID)


@lru_cache(maxsize=1)
def get_embedding_vocab() -> EmbeddingVocab:
    payload = json.loads(VOCAB_PATH.read_text(encoding="utf-8"))
    return EmbeddingVocab(payload)


def vocab_sizes() -> Dict[str, int]:
    vocab = get_embedding_vocab()
    return {
        "species_vocab_size": vocab.species_vocab_size,
        "item_vocab_size": vocab.item_vocab_size,
        "ability_vocab_size": vocab.ability_vocab_size,
    }
