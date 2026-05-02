# Reward Function Audit TODOs

## 1) Prevent reward spikes when curriculum changes reward config mid-episode

- **Problem:** `set_reward_config()` updates coefficients immediately, but `_reward_buffer` still holds a value computed with the old config.
- **Impact:** Next reward can include an artificial jump not caused by the latest action.
- **Action options:**
  - Rebase `_reward_buffer[battle]` when reward config changes.
  - Or apply reward config changes only at episode reset boundaries.

## 2) Use `fainted_penalty` for our fainted Pokemon

- **Problem:** Live reward path uses `-fainted_value` for our faints instead of `fainted_penalty`.
- **Impact:** Tuning `fainted_penalty` in `RewardConfig` has no effect on training reward.
- **Action:** In `_compute_configured_delta_reward()`, replace own-faint handling to use `self.reward_config.fainted_penalty`.

## 3) Include `step_penalty` in training reward (if intended)

- **Problem:** `step_penalty` is defined in config and used in terminal diagnostics, but not included in returned per-step reward.
- **Impact:** No explicit pressure toward shorter battles in actual PPO learning signal.
- **Action:** Add `step_penalty` to the live reward calculation (per step or turn-delta based, depending on intended shaping).

## 4) Remove hardcoded team size assumption (`6`)

- **Problem:** Reward logic assumes 6v6 (`number_of_pokemons = 6`).
- **Impact:** Reward shaping can be incorrect in non-6v6/custom formats.
- **Action:** Replace hardcoded value with a format-derived or config-driven team size.

## 5) Support fixed-team training until a target win rate

- **Goal:** Keep a fixed player team (and optionally fixed opponent team) during early curriculum, then relax once win-rate threshold is reached.
- **Action options:**
  - Add stage-level fields to curriculum payload (for example: `fixed_player_team`, `fixed_opponent_team`, `unlock_at_win_rate`).
  - In curriculum transition logic, keep these fields active until rolling WR meets threshold, then clear them in a later stage.
  - Ensure wrapper reset uses currently active team settings when constructing/sampling opponents so the switch happens at episode boundaries.
- **Validation:** Log active team mode per episode and verify it flips only after threshold is met.

## Suggested validation after fixes

- Run a short training smoke test: `uv run train_battler.py --preset quick`.
- Check that:
  - reward does not jump at curriculum stage transition without state change,
  - `fainted_penalty` sensitivity is visible when changed,
  - `step_penalty` affects average episode length as expected.
