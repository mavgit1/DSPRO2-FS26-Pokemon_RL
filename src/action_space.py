from __future__ import annotations

from typing import List

import numpy as np
from poke_env.battle.abstract_battle import AbstractBattle
from poke_env.environment.singles_env import SinglesEnv


NATIVE_ACTION_SPACE_N = 22
COMPRESSED_ACTION_SPACE_N = 14

COMPRESSED_MOVE_ACTIONS = range(0, 4)
COMPRESSED_GIMMICK_ACTIONS = range(4, 8)
COMPRESSED_SWITCH_ACTIONS = range(8, 14)

NATIVE_SWITCH_ACTIONS = range(0, 6)
NATIVE_GIMMICK_OFFSETS = (10, 14, 18)


def compressed_to_native_action(action: int, battle: AbstractBattle) -> np.int64:
    """Map a compressed agent action to a legal native poke-env action."""
    action_int = int(action)

    if action_int in COMPRESSED_MOVE_ACTIONS:
        return np.int64(6 + action_int)
    if action_int in COMPRESSED_GIMMICK_ACTIONS:
        return _compressed_gimmick_to_native(action_int, battle)
    if action_int in COMPRESSED_SWITCH_ACTIONS:
        return _compressed_switch_to_native(action_int, battle)

    raise ValueError(f"Compressed action out of range: {action_int}")


def get_compressed_action_mask(battle: AbstractBattle) -> np.ndarray:
    """Return a compressed action mask using native poke-env legality."""
    mask = np.zeros(COMPRESSED_ACTION_SPACE_N, dtype=np.float32)
    for action in range(COMPRESSED_ACTION_SPACE_N):
        try:
            native_action = compressed_to_native_action(action, battle)
            _verify_native_action(native_action, battle)
            mask[action] = 1.0
        except (IndexError, ValueError):
            continue

    if not mask.any():
        fallback = _first_legal_native_action(battle)
        if fallback is not None:
            compressed = native_to_compressed_action(int(fallback), battle)
            if compressed is not None:
                mask[compressed] = 1.0
        if not mask.any():
            mask[0] = 1.0
    return mask


def native_to_compressed_action(
    native_action: int,
    battle: AbstractBattle,
) -> int | None:
    """Best-effort native-to-compressed conversion for fallback diagnostics."""
    native_int = int(native_action)
    if 6 <= native_int <= 9:
        return native_int - 6
    for offset in NATIVE_GIMMICK_OFFSETS:
        if offset <= native_int <= offset + 3:
            move_slot = native_int - offset
            compressed = COMPRESSED_GIMMICK_ACTIONS.start + move_slot
            try:
                mapped = compressed_to_native_action(compressed, battle)
            except ValueError:
                return None
            if int(mapped) == native_int:
                return compressed
            return None
    if native_int in NATIVE_SWITCH_ACTIONS:
        bench_native = _bench_native_actions(battle)
        try:
            bench_idx = bench_native.index(native_int)
        except ValueError:
            return None
        if bench_idx < len(COMPRESSED_SWITCH_ACTIONS):
            return COMPRESSED_SWITCH_ACTIONS.start + bench_idx
    return None


def is_compressed_switch_action(action: int) -> bool:
    return int(action) in COMPRESSED_SWITCH_ACTIONS


def _compressed_gimmick_to_native(action: int, battle: AbstractBattle) -> np.int64:
    move_slot = int(action) - COMPRESSED_GIMMICK_ACTIONS.start
    if move_slot < 0 or move_slot >= 4:
        raise ValueError(f"No legal native gimmick for compressed action {action}")
    for offset in NATIVE_GIMMICK_OFFSETS:
        native_action = np.int64(offset + move_slot)
        try:
            _verify_native_action(native_action, battle)
            return native_action
        except (IndexError, ValueError):
            continue
    raise ValueError(f"No legal native gimmick for compressed action {action}")


def _bench_native_actions(battle: AbstractBattle) -> List[int]:
    """Native action indices for non-active team members in team order.

    Matches the observation layout where bench tokens 2-6 contain
    non-active team members in ``battle.team.values()`` order, so
    compressed switch 8+k always maps to bench token 2+k.
    """
    team_list = list(battle.team.values())
    active = battle.active_pokemon
    return [i for i, mon in enumerate(team_list) if mon is not active]


def _compressed_switch_to_native(action: int, battle: AbstractBattle) -> np.int64:
    bench_native = _bench_native_actions(battle)
    switch_idx = int(action) - COMPRESSED_SWITCH_ACTIONS.start
    if switch_idx < 0 or switch_idx >= len(bench_native):
        raise ValueError(f"No bench slot for compressed switch action {action}")
    return np.int64(bench_native[switch_idx])


def _legal_native_switch_actions(battle: AbstractBattle) -> List[int]:
    legal = []
    for native_action in NATIVE_SWITCH_ACTIONS:
        try:
            _verify_native_action(np.int64(native_action), battle)
            legal.append(int(native_action))
        except (IndexError, ValueError):
            continue
    return legal


def _first_legal_native_action(battle: AbstractBattle) -> int | None:
    for native_action in range(NATIVE_ACTION_SPACE_N):
        try:
            _verify_native_action(np.int64(native_action), battle)
            return native_action
        except (IndexError, ValueError):
            continue
    return None


def _verify_native_action(native_action: np.int64, battle: AbstractBattle) -> None:
    SinglesEnv.action_to_order(
        native_action,
        battle,
        fake=False,
        strict=True,
    )
