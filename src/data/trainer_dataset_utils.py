#!/usr/bin/env python3
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

from scripts.validate_bdsp_trainers import validate_dataset

def load_trainers(json_path: str | Path, strict: bool = False, allow_known_source_errors: bool = True) -> dict[str, Any]:
    path = Path(json_path)
    with path.open('r', encoding='utf-8') as f:
        data = json.load(f)

    report = validate_dataset(data, strict=strict)
    errors = report['errors']

    if allow_known_source_errors:
        errors = [e for e in errors if 'EV sum is 514' not in e]

    if errors:
        error_text = '\n'.join(errors[:20])
        raise ValueError(f'Trainer dataset failed validation:\n{error_text}')
    return data

def pokemon_to_showdown(mon: dict[str, Any]) -> str:
    lines: list[str] = []
    first = mon['species']
    if mon.get('item'):
        first += f" @ {mon['item']}"
    lines.append(first)
    if mon.get('ability'):
        lines.append(f"Ability: {mon['ability']}")
    lines.append(f"Level: {mon['level']}")
    if mon.get('nature'):
        lines.append(f"{mon['nature']} Nature")

    ev_parts = []
    for pretty, key in [('HP', 'hp'), ('Atk', 'atk'), ('Def', 'def'), ('SpA', 'spa'), ('SpD', 'spd'), ('Spe', 'spe')]:
        value = mon['evs'].get(key, 0)
        if value:
            ev_parts.append(f'{value} {pretty}')
    if ev_parts:
        lines.append('EVs: ' + ' / '.join(ev_parts))

    iv_parts = []
    for pretty, key in [('HP', 'hp'), ('Atk', 'atk'), ('Def', 'def'), ('SpA', 'spa'), ('SpD', 'spd'), ('Spe', 'spe')]:
        value = mon['ivs'].get(key, 31)
        if value != 31:
            iv_parts.append(f'{value} {pretty}')
    if iv_parts:
        lines.append('IVs: ' + ' / '.join(iv_parts))

    for move in mon.get('moves', []):
        lines.append(f'- {move}')

    return '\n'.join(lines)

def trainer_to_showdown_team(trainer: dict[str, Any]) -> str:
    return '\n\n'.join(pokemon_to_showdown(mon) for mon in trainer['party'])

def sample_random_team(json_path: str | Path, seed: int | None = None):
    rng = random.Random(seed)
    data = load_trainers(json_path)
    trainer = rng.choice(data['trainers'])
    return trainer, trainer_to_showdown_team(trainer)
