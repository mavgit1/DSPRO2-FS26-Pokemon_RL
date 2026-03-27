#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

STAT_KEYS = {'hp', 'atk', 'def', 'spa', 'spd', 'spe'}

def load_json(path: Path) -> dict[str, Any]:
    with path.open('r', encoding='utf-8') as f:
        return json.load(f)

def validate_dataset(data: dict[str, Any], strict: bool = False) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []

    if not isinstance(data, dict):
        return {'ok': False, 'errors': ['Top-level JSON must be an object.'], 'warnings': [], 'summary': {}}

    meta = data.get('meta')
    trainers = data.get('trainers')

    if not isinstance(meta, dict):
        errors.append("Missing or invalid 'meta' object.")
    if not isinstance(trainers, list):
        errors.append("Missing or invalid 'trainers' list.")
        return {'ok': False, 'errors': errors, 'warnings': warnings, 'summary': {}}

    seen_ids: set[int] = set()
    total_pokemon = 0

    for ti, trainer in enumerate(trainers, start=1):
        prefix = f'trainer[{ti}]'
        if not isinstance(trainer, dict):
            errors.append(f'{prefix}: trainer must be an object.')
            continue

        trainer_id = trainer.get('trainer_id')
        trainer_label = trainer.get('trainer_label')
        party_size = trainer.get('party_size')
        party = trainer.get('party')

        if not isinstance(trainer_id, int) or trainer_id <= 0:
            errors.append(f'{prefix}: trainer_id must be a positive integer.')
        elif trainer_id in seen_ids:
            errors.append(f'{prefix}: duplicate trainer_id={trainer_id}.')
        else:
            seen_ids.add(trainer_id)

        if not isinstance(trainer_label, str) or not trainer_label.strip():
            errors.append(f'{prefix}: trainer_label must be a non-empty string.')

        if not isinstance(party, list):
            errors.append(f'{prefix}: party must be a list.')
            continue

        if len(party) == 0:
            errors.append(f'{prefix}: party is empty.')
        if len(party) > 6:
            errors.append(f'{prefix}: party has {len(party)} Pokémon, expected at most 6.')
        if party_size != len(party):
            errors.append(f'{prefix}: party_size={party_size} does not match len(party)={len(party)}.')

        for pi, mon in enumerate(party, start=1):
            mp = f'{prefix}.party[{pi}]'
            total_pokemon += 1

            if not isinstance(mon, dict):
                errors.append(f'{mp}: Pokémon must be an object.')
                continue

            slot = mon.get('slot')
            species = mon.get('species')
            level = mon.get('level')
            sticker = mon.get('sticker')
            ability = mon.get('ability')
            item = mon.get('item')
            moves = mon.get('moves')
            nature = mon.get('nature')
            evs = mon.get('evs')
            ivs = mon.get('ivs')

            if slot != pi:
                errors.append(f'{mp}: slot={slot} expected {pi}.')
            if not isinstance(species, str) or not species.strip():
                errors.append(f'{mp}: species must be a non-empty string.')
            if not isinstance(level, int) or not (1 <= level <= 100):
                errors.append(f'{mp}: level must be an integer in [1, 100].')
            if sticker is not None and not isinstance(sticker, str):
                errors.append(f'{mp}: sticker must be null or a string.')
            if ability is None:
                warnings.append(f'{mp}: ability is missing.')
            elif not isinstance(ability, str):
                errors.append(f'{mp}: ability must be null or a string.')
            if item is not None and not isinstance(item, str):
                errors.append(f'{mp}: item must be null or a string.')
            if nature is None:
                warnings.append(f'{mp}: nature is missing.')
            elif not isinstance(nature, str):
                errors.append(f'{mp}: nature must be null or a string.')
            if not isinstance(moves, list):
                errors.append(f'{mp}: moves must be a list.')
            else:
                if len(moves) > 4:
                    errors.append(f'{mp}: moves has {len(moves)} entries, expected at most 4.')
                if len(moves) == 0:
                    msg = f'{mp}: moves list is empty.'
                    if strict:
                        errors.append(msg)
                    else:
                        warnings.append(msg)
                for mi, move in enumerate(moves, start=1):
                    if not isinstance(move, str) or not move.strip():
                        errors.append(f'{mp}: move[{mi}] must be a non-empty string.')

            for stat_name, stat_block, upper in (('evs', evs, 252), ('ivs', ivs, 31)):
                if not isinstance(stat_block, dict):
                    errors.append(f'{mp}: {stat_name} must be an object.')
                    continue
                if set(stat_block.keys()) != STAT_KEYS:
                    errors.append(f'{mp}: {stat_name} keys must be exactly {sorted(STAT_KEYS)}.')
                    continue
                total = 0
                for key, value in stat_block.items():
                    if not isinstance(value, int):
                        errors.append(f'{mp}: {stat_name}.{key} must be an integer.')
                        continue
                    if not (0 <= value <= upper):
                        errors.append(f'{mp}: {stat_name}.{key}={value} out of range 0..{upper}.')
                    total += value
                if stat_name == 'evs' and total > 510:
                    errors.append(f'{mp}: EV sum is {total}, expected <= 510.')

    return {
        'ok': len(errors) == 0,
        'errors': errors,
        'warnings': warnings,
        'summary': {
            'trainer_count': len(trainers),
            'pokemon_count': total_pokemon,
            'error_count': len(errors),
            'warning_count': len(warnings),
        },
    }

def main() -> int:
    parser = argparse.ArgumentParser(description='Validate BDSP trainer dataset JSON.')
    parser.add_argument('json_path', type=Path, help='Path to bdsp_trainers.json')
    parser.add_argument('--strict', action='store_true', help='Treat empty move lists as errors.')
    args = parser.parse_args()

    data = load_json(args.json_path)
    report = validate_dataset(data, strict=args.strict)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if report['ok'] else 1

if __name__ == '__main__':
    raise SystemExit(main())
