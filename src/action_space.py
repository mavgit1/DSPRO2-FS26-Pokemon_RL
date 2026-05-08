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
    """Return a compressed action mask based on direct battle state.

    Uses ``battle.available_moves`` and ``battle.available_switches``
    directly instead of calling ``action_to_order`` with ``strict=True``,
    which can fail when ``battle.valid_orders`` is stale or empty.

    This avoids the previous fallback that forced ``mask[0] = 1.0`` when
    no action passed verification — a fallback that contaminated ~14% of
    training samples with incorrect action labels.
    """
    mask = np.zeros(COMPRESSED_ACTION_SPACE_N, dtype=np.float32)

    active = battle.active_pokemon
    force_switch = getattr(battle, "force_switch", False)
    trapped = getattr(battle, "trapped", False)

    # --- Switch actions (compressed 8-13) ---
    if not trapped:
        _mark_available_switches(mask, battle)

    # --- Move actions (compressed 0-3) ---
    if active is not None and not force_switch:
        _mark_available_moves(mask, battle, active)

    # --- Gimmick actions (compressed 4-7) ---
    # Gimmicks are rare (gen6+ only); existing verification is acceptable.
    if active is not None and not force_switch:
        for gim_i in range(len(COMPRESSED_GIMMICK_ACTIONS)):
            try:
                _compressed_gimmick_to_native(
                    COMPRESSED_GIMMICK_ACTIONS.start + gim_i, battle
                )
                mask[COMPRESSED_GIMMICK_ACTIONS.start + gim_i] = 1.0
            except (IndexError, ValueError):
                continue

    # Safety net: when the mask is empty (transition states like post-faint,
    # battle init), find any valid action via find_safe_native_action.
    # This is much rarer than the old mask[0]=1.0 fallback which triggered
    # ~14% of the time due to stale valid_orders.
    if not mask.any():
        safe = find_safe_native_action(battle)
        compressed = native_to_compressed_action(int(safe), battle)
        if compressed is not None:
            mask[compressed] = 1.0

    # If still empty (e.g. safe native -2 unmappable to compressed), recover with
    # strict legality checks — rare; avoids an all-zero mask breaking the policy.
    if not mask.any():
        _fill_mask_from_strict_verify(mask, battle)
    if not mask.any():
        mask[0] = 1.0

    return mask


def _mark_available_moves(mask: np.ndarray, battle: AbstractBattle, active) -> None:
    available_moves = getattr(battle, "available_moves", [])

    # Struggle / recharge: only move slot 0 is valid
    if len(available_moves) == 1 and available_moves[0].id in ("struggle", "recharge"):
        mask[0] = 1.0
        return

    # Normal case: map available move IDs to their slot indices
    available_ids = {m.id for m in available_moves}
    known_moves = list(active.moves.values())
    for i, move in enumerate(known_moves):
        if i >= 4:
            break
        if move.id in available_ids:
            mask[i] = 1.0


def _mark_available_switches(mask: np.ndarray, battle: AbstractBattle) -> None:
    available_switches = {id(mon) for mon in getattr(battle, "available_switches", [])}
    if not available_switches:
        return
    bench_native = _bench_native_actions(battle)
    team_list = list(battle.team.values())
    for k, native_idx in enumerate(bench_native):
        if k >= len(COMPRESSED_SWITCH_ACTIONS):
            break
        if (
            native_idx < len(team_list)
            and id(team_list[native_idx]) in available_switches
        ):
            mask[COMPRESSED_SWITCH_ACTIONS.start + k] = 1.0


def find_safe_native_action(battle: AbstractBattle) -> np.int64:
    """Find a guaranteed-valid native action using direct availability.

    Used as a safety net when the normal compressed→native conversion
    produces an action that poke-env rejects (e.g. stale ``valid_orders``).
    """
    available_moves = getattr(battle, "available_moves", [])
    available_switches = getattr(battle, "available_switches", [])
    active = battle.active_pokemon
    force_switch = getattr(battle, "force_switch", False)

    # Prefer a move if not forced to switch
    if not force_switch and active is not None and available_moves:
        # Struggle / recharge → native action 6 (move slot 0)
        if len(available_moves) == 1 and available_moves[0].id in (
            "struggle",
            "recharge",
        ):
            return np.int64(6)

        known_moves = list(active.moves.values())
        available_ids = {m.id for m in available_moves}
        for i, move in enumerate(known_moves):
            if i >= 4:
                break
            if move.id in available_ids:
                return np.int64(6 + i)

    # Fallback to a switch
    if available_switches:
        switch_set = set(available_switches)
        for i, mon in enumerate(battle.team.values()):
            if mon in switch_set:
                return np.int64(i)

    # Last resort: default (let poke-env handle it)
    return np.int64(-2)


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


def _fill_mask_from_strict_verify(mask: np.ndarray, battle: AbstractBattle) -> None:
    """Enable the first compressed index that passes SinglesEnv strict legality."""
    for a in range(COMPRESSED_ACTION_SPACE_N):
        try:
            nat = compressed_to_native_action(a, battle)
            _verify_native_action(nat, battle)
            mask[a] = 1.0
            return
        except (ValueError, IndexError, TypeError):
            continue


def _verify_native_action(native_action: np.int64, battle: AbstractBattle) -> None:
    SinglesEnv.action_to_order(
        native_action,
        battle,
        fake=False,
        strict=True,
    )
