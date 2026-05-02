#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path

ELITE_FOUR_NAMES = ("aaron", "bertha", "flint", "lucian")
CHAMPION_NAMES = ("cynthia",)
GYM_LEADER_NAMES = (
    "roark",
    "gardenia",
    "maylene",
    "crasher wake",
    "fantina",
    "byron",
    "candice",
    "volkner",
)


def _split_csv_arg(value: str) -> list[str]:
    if not value.strip():
        return []
    return [part.strip() for part in value.split(",") if part.strip()]


def _normalize_name(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip().lower())


def _contains_any(name: str, candidates: tuple[str, ...]) -> bool:
    normalized = _normalize_name(name)
    return any(candidate in normalized for candidate in candidates)


def _is_override_match(trainer_name: str, trainer_id: int, token: str) -> bool:
    value = token.strip()
    if not value:
        return False
    if value.isdigit():
        return int(value) == trainer_id
    return _normalize_name(value) == _normalize_name(trainer_name)


def _max_level(trainer: dict) -> int:
    levels = [
        int(mon.get("level"))
        for mon in trainer.get("party", [])
        if isinstance(mon.get("level"), int)
    ]
    return max(levels) if levels else -1


def _avg_level(trainer: dict) -> float:
    levels = [
        int(mon.get("level"))
        for mon in trainer.get("party", [])
        if isinstance(mon.get("level"), int)
    ]
    return (sum(levels) / len(levels)) if levels else -1.0


def _breakpoint_for(trainer_name: str, trainer_id: int, overrides: dict[str, list[str]]) -> str:
    is_elite = _contains_any(trainer_name, ELITE_FOUR_NAMES) or any(
        _is_override_match(trainer_name, trainer_id, token)
        for token in overrides["elite_four"]
    )
    is_champion = _contains_any(trainer_name, CHAMPION_NAMES) or any(
        _is_override_match(trainer_name, trainer_id, token)
        for token in overrides["champion"]
    )
    is_gym = _contains_any(trainer_name, GYM_LEADER_NAMES) or any(
        _is_override_match(trainer_name, trainer_id, token)
        for token in overrides["gym_leader"]
    )

    if is_champion:
        return "champion"
    if is_elite:
        return "elite_four"
    if is_gym:
        return "gym_leader"
    return "none"


def _validate_breakpoint_constraints(entries: list[dict], strict: bool) -> tuple[bool, list[str]]:
    problems: list[str] = []
    elite_idxs = [entry["order_index"] for entry in entries if entry["breakpoint"] == "elite_four"]
    champ_idxs = [entry["order_index"] for entry in entries if entry["breakpoint"] == "champion"]

    if elite_idxs:
        expected = list(range(min(elite_idxs), max(elite_idxs) + 1))
        if elite_idxs != expected:
            problems.append("Elite Four trainers are not contiguous in gauntlet order.")

    if champ_idxs and elite_idxs and min(champ_idxs) != max(elite_idxs) + 1:
        problems.append("Champion does not immediately follow the Elite Four block.")

    return ((not strict) or (len(problems) == 0)), problems


def main() -> None:
    parser = argparse.ArgumentParser(description="One-time generation of ordered BDSP gauntlet JSON.")
    parser.add_argument(
        "--input-path",
        type=Path,
        default=Path("data/bdsp_trainers.json"),
        help="Path to canonical trainer JSON source.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("data/bdsp_gauntlet_order.json"),
        help="Path for ordered gauntlet JSON output.",
    )
    parser.add_argument(
        "--elite-four-overrides",
        type=str,
        default="",
        help="Comma-separated trainer IDs or names to force as Elite Four.",
    )
    parser.add_argument(
        "--champion-overrides",
        type=str,
        default="",
        help="Comma-separated trainer IDs or names to force as Champion.",
    )
    parser.add_argument(
        "--gym-leader-overrides",
        type=str,
        default="",
        help="Comma-separated trainer IDs or names to force as Gym Leader.",
    )
    parser.add_argument(
        "--strict-breakpoints",
        action="store_true",
        help="Fail if Elite Four and Champion constraints are not met.",
    )
    args = parser.parse_args()

    payload = json.loads(args.input_path.read_text(encoding="utf-8"))
    trainers = payload.get("trainers", [])

    overrides = {
        "elite_four": _split_csv_arg(args.elite_four_overrides),
        "champion": _split_csv_arg(args.champion_overrides),
        "gym_leader": _split_csv_arg(args.gym_leader_overrides),
    }

    rows = []
    for trainer in trainers:
        trainer_id = int(trainer.get("trainer_id"))
        trainer_name = str(trainer.get("trainer_label", f"Trainer {trainer_id}"))
        rows.append(
            {
                "trainer_id": trainer_id,
                "trainer_name": trainer_name,
                "max_level": _max_level(trainer),
                "avg_level": _avg_level(trainer),
                "team_size": int(trainer.get("party_size", len(trainer.get("party", [])))),
                "breakpoint": _breakpoint_for(trainer_name, trainer_id, overrides),
            }
        )

    regular = [row for row in rows if row["breakpoint"] not in {"elite_four", "champion"}]
    elite = [row for row in rows if row["breakpoint"] == "elite_four"]
    champion = [row for row in rows if row["breakpoint"] == "champion"]

    ordered_rows = sorted(regular, key=lambda r: (r["max_level"], r["avg_level"], r["trainer_id"]))
    ordered_rows += sorted(elite, key=lambda r: (r["max_level"], r["avg_level"], r["trainer_id"]))
    ordered_rows += sorted(champion, key=lambda r: (r["max_level"], r["avg_level"], r["trainer_id"]))

    entries = []
    for idx, row in enumerate(ordered_rows):
        row["order_index"] = idx
        entries.append(row)

    is_valid, problems = _validate_breakpoint_constraints(entries, strict=args.strict_breakpoints)

    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_path": str(args.input_path),
        "trainer_count": len(entries),
        "strict_breakpoints": args.strict_breakpoints,
        "is_valid": is_valid,
        "problems": problems,
    }
    output_payload = {"metadata": metadata, "entries": entries}
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    args.output_path.write_text(json.dumps(output_payload, indent=2), encoding="utf-8")

    print(f"Wrote gauntlet order to {args.output_path}")
    print(f"Trainers parsed: {len(entries)}")
    if problems:
        print("Constraint warnings:")
        for item in problems:
            print(f" - {item}")
    if args.strict_breakpoints and not is_valid:
        raise SystemExit(2)


if __name__ == "__main__":
    main()

