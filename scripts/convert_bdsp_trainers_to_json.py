#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import re
from pathlib import Path

STAT_KEYS = ['hp', 'atk', 'def', 'spa', 'spd', 'spe']
TRAINER_RE = re.compile(r'^Trainer ID (\d+)$')

def norm(value):
    if value is None:
        return None
    s = str(value).strip()
    if not s or s.lower() == 'nan' or s == '--':
        return None
    return s

def as_int(value):
    s = norm(value)
    if s is None:
        return None
    if re.fullmatch(r'-?\d+(\.0+)?', s):
        return int(float(s))
    return None

def convert(csv_path: Path) -> dict:
    rows = []
    with csv_path.open('r', encoding='utf-8-sig', newline='') as f:
        reader = csv.reader(f)
        for row in reader:
            rows.append(row)

    width = max(len(r) for r in rows)
    rows = [r + [None] * (width - len(r)) for r in rows]

    trainers = []
    current = None
    for idx, row in enumerate(rows, start=1):
        c1 = norm(row[1] if len(row) > 1 else None)
        if c1 == 'Pokémon':
            continue
        match = TRAINER_RE.match(c1 or '')
        if match:
            current = {
                'trainer_id': int(match.group(1)),
                'trainer_label': norm(row[3] if len(row) > 3 else None),
                'source_row': idx,
                'party': [],
            }
            trainers.append(current)
            continue
        if current is None or not c1:
            continue
        moves_raw = norm(row[7] if len(row) > 7 else None)
        current['party'].append({
            'slot': len(current['party']) + 1,
            'source_row': idx,
            'species': c1,
            'sticker': norm(row[2] if len(row) > 2 else None),
            'level': as_int(row[3] if len(row) > 3 else None),
            'ability': norm(row[4] if len(row) > 4 else None) or norm(row[5] if len(row) > 5 else None),
            'item': norm(row[6] if len(row) > 6 else None),
            'moves': [m.strip() for m in moves_raw.split('/') if m.strip()] if moves_raw else [],
            'nature': norm(row[8] if len(row) > 8 else None),
            'evs': {k: (as_int(row[c]) if as_int(row[c]) is not None else 0) for k, c in zip(STAT_KEYS, range(9, 15))},
            'ivs': {k: (as_int(row[c]) if as_int(row[c]) is not None else 0) for k, c in zip(STAT_KEYS, range(15, 21))},
        })

    for trainer in trainers:
        trainer['party_size'] = len(trainer['party'])

    return {
        'meta': {
            'dataset_name': 'BDSP Trainer Data',
            'source_csv': csv_path.name,
            'schema_version': '1.0.0',
            'generated_at_utc': dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat().replace('+00:00', 'Z'),
            'trainer_count': len(trainers),
            'pokemon_count': sum(len(t['party']) for t in trainers),
            'notes': [
                'Converted from a Google Sheets CSV export.',
                "String placeholder '--' was normalized to null.",
                'Ability may appear in one of two neighboring columns due to sheet layout.'
            ],
        },
        'trainers': trainers,
    }

def main() -> int:
    parser = argparse.ArgumentParser(description='Convert BDSP trainer CSV to structured JSON.')
    parser.add_argument('csv_path', type=Path, help='Path to the exported CSV file.')
    parser.add_argument('--output', type=Path, default=Path('bdsp_trainers.json'), help='Output JSON path.')
    args = parser.parse_args()
    dataset = convert(args.csv_path)
    args.output.write_text(json.dumps(dataset, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f"Wrote {args.output} with {dataset['meta']['trainer_count']} trainers and {dataset['meta']['pokemon_count']} Pokémon.")
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
