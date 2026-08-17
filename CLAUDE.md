# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Reinforcement learning project training PPO agents to play Pokémon battles (BDSP/Nuzlocke). Uses Ray RLlib for distributed training, PyTorch for neural networks, `poke-env` as the Gymnasium-compatible interface to local Pokémon Showdown servers, and MLflow for experiment tracking.

## Common Commands

```bash
# Install dependencies (uses uv, not pip)
uv sync

# Start/stop Pokémon Showdown servers (training also starts any missing ports)
./scripts/spin_up_multiple_showdown.sh    # spins up 8 servers on ports 8000-8007
./scripts/kill_all_showdown.sh

# Set up custom Showdown formats (no Dynamax/Tera, no Sleep Clause)
# Run once, then restart servers. Creates pokemon-showdown/config/custom-formats.ts
./scripts/setup_custom_formats.sh

# One-shot setup (clean install) — runs all of the above
./scripts/setup_training.sh

# Training (use the venv interpreter — do NOT `uv run` Ray jobs; workers inherit uv and re-download Torch)
.venv/bin/python train_battler.py --preset quick
.venv/bin/python train_battler.py --preset standard
.venv/bin/python train_battler.py --preset optimal          # RTX 5090
.venv/bin/python train_battler.py --preset memory_safe
.venv/bin/python train_battler.py --preset large
.venv/bin/python train_battler.py --preset pure_league_play

# Resume training from checkpoint
.venv/bin/python train_battler.py --preset optimal --resume-checkpoint latest --mlflow-run-id <RUN_ID>

# View local MLflow runs
mlflow ui --backend-store-uri sqlite:///mlflow.db

# Validation (benchmark: 3 opponents x N episodes)
.venv/bin/python scripts/validate_checkpoint.py --checkpoint checkpoints/step_XXXXXX --protocol benchmark --preset standard
.venv/bin/python scripts/validate_checkpoint.py --checkpoint checkpoints/step_XXXXXX --protocol benchmark --preset standard --explore   # stochastic policy
.venv/bin/python scripts/validate_checkpoint.py --checkpoint checkpoints/step_XXXXXX --protocol smoke --preset quick                   # quick 3-episode check

# Self-play diagnostics run (30% self-play, never promotes from stage 0)
.venv/bin/python scripts/diagnose_selfplay.py --preset standard --timesteps 500000

# Hyperparameter sweep (Optuna TPE, 500k steps/trial, resumes from SQLite)
.venv/bin/python scripts/hparam_sweep.py --n-trials 50                                     # full sweep (~14h)
.venv/bin/python scripts/hparam_sweep.py --n-trials 3 --timesteps 100000                   # dry run

# Profile one PPO iteration (Showdown must already be running)
.venv/bin/python scripts/profile_training_iteration.py --preset quick --iterations 3 --num-servers 8

# Lint/format
uv run ruff check .
uv run ruff format .

# Dependencies
uv add <package>       # add dependency
uv remove <package>    # remove dependency
uv cache clean         # clean cache periodically
```

## Architecture

**Entry point**: `train_battler.py` — CLI that parses args, loads a config preset, and calls `PokemonTrainer.train()`.

**Training pipeline** (`src/training/`):
- `trainer.py` (`PokemonTrainer`): Orchestrates the full lifecycle — config building, environment registration, PPO algorithm creation, training loop, MLflow logging, and checkpointing.
- `rllib_config_builder.py`: Constructs the RLlib PPO config and registers the custom Gymnasium environment.
- `env_bridge.py`: Worker-side bridge that injects curriculum payloads into environments and collects env-emitted metrics.
- `callbacks.py`: RLlib callbacks for curriculum stage progression and checkpoint management.
- `curriculum.py`: Defines progressive difficulty scaling across training.
- `resume.py`: Checkpoint path resolution and step extraction for interrupted runs.
- `metrics/`: Metric extraction and aggregation (PPO, episode, runtime, flattening).
- `monitoring/`: Runtime system telemetry (CPU/RAM/GPU).

**Model** (`src/models/`):
- `battle_transformer.py`: Transformer-based architecture with species/item/ability embeddings, optional LSTM for temporal memory, and action masking for valid moves only.
- `embedding.py`: Embedding layers for Pokémon battle entities; includes global extras such as opponent-type bits (see opponent-key note under Environment).

**Environment** (`src/envs/`):
- `battle_env.py`: Custom Gymnasium environment wrapping `poke-env`. Connects to local Pokémon Showdown servers, defines observation/action spaces, and handles battle state representation.
- `random_no_switch_player.py`: Custom poke-env `Player` that samples uniformly among legal **move** orders (including mega / Z / Dynamax / Tera variants) and **never voluntarily switches**; it still picks randomly among legal switches when Showdown forces a switch or no moves are available.

**Opponent keys** *(documented 2026-05-04)*:
- **`random`**: poke-env `RandomPlayer` — uniform over all `battle.valid_orders` (switches and moves).
- **`random_no_switch`**: `RandomNoSwitchPlayer` — same as above except switches are excluded whenever at least one move order exists. Use `opponent_difficulty="random_no_switch"` or include `"random_no_switch"` in curriculum `opponent_mix` (see `create_env_creator` in `battle_env.py`).
- **`heuristic`**: `SimpleHeuristicsPlayer`. **`self`**: `SelfPlayPlayer` with exported weights.
- **Embeddings**: In `embedding.py`, `_canonical_opponent_type` maps `random_no_switch` to `random`, so the global “opponent type” feature vector treats both as the same bucket (first flag), avoiding a distribution shift at the observation layer when mixing or swapping baselines.

**Configuration** (`src/config/`):
- `TM_optimal_config.py`: Dataclass-based config with 5 hardware presets (quick, standard, memory_safe, optimal, large). Each preset defines model architecture, PPO hyperparameters, environment parallelism, and curriculum parameters.

**Data** (`src/data/`, `src/teams/`): Trainer dataset utilities and team generation for BDSP opponents.

**Data files** (`data/`): BDSP trainer CSV, gauntlet order JSON, trainer teams JSON, validation team manifests.

**Validation** (`src/validation/`):
- `protocols.py`: Protocol definitions. `benchmark` runs 3 opponent tiers (random, random_no_switch, heuristic) × N episodes. Other protocols: `smoke`, `fixed_paired`, `mirror`. `gauntlet_first_loss` removed.
- `runner.py`: `run_validation()` for subprocess-based checkpoint evaluation; `run_inprocess_validation()` for live algo evaluation during training (no Ray init/restore overhead). Supports `--explore` for stochastic action sampling via masked softmax.
- `metrics.py`: Per-episode `BattleResult`, `aggregate_validation_metrics`, `compute_benchmark_metrics` (per-opponent WR with Wilson CIs, skill score, consistency).
- `reporting.py`: MLflow logging, JSON report writing, `format_validation_summary()` for console output.
- `teams.py`: Battle spec generation for fixed-paired and mirror protocols.

**Fixed player team** *(added 2026-05-05)*:
- Set `EnvironmentConfig.player_team_path` to a Showdown-format `.txt` team file (default: `data/teams/player_team.txt`).
- When set, the RL agent always uses that team. The battle format auto-switches from `*randombattle` to `*customgame` (generation is preserved: `gen8randombattle` → `gen8customgame`).
- Works with `*nogimmicks` variants too: `gen8randombattlenogimmicks` → `gen8customgamenogimmicks`.
- **Team must match the generation** — a gen8 team in a gen5 battle will fail on Showdown. Set `battle_format` to match the team's generation.
- Validation: `--player-team <path>` flag. Benchmark protocol uses the team against all opponent tiers. Mirror protocol uses the same team for both sides. Fixed-paired uses it for the RL side only.
- The team is logged to MLflow as a `player_team.txt` artifact.

**Custom Showdown formats** *(added 2026-05-05)*:
- `scripts/setup_custom_formats.sh` writes `pokemon-showdown/config/custom-formats.ts` with no-gimmick format variants.
- Available formats:
  - `[Gen 8] Random Battle (No Gimmicks)` — `gen8randombattlenogimmicks` — no Dynamax, no Sleep Clause, random teams
  - `[Gen 9] Random Battle (No Gimmicks)` — `gen9randombattlenogimmicks` — no Terastallize, no Sleep Clause, random teams
  - `[Gen 8] Custom Game (No Gimmicks)` — `gen8customgamenogimmicks` — no Dynamax, fixed teams
  - `[Gen 9] Custom Game (No Gimmicks)` — `gen9customgamenogimmicks` — no Terastallize, fixed teams
- **Must restart Showdown servers** after running the setup script.
- Custom Game variants list rules explicitly (not inherited) because the base Custom Game doesn't include `Sleep Clause Mod` — trying to remove it with `!Sleep Clause Mod` crashes the server.
- To use: set `EnvironmentConfig.battle_format` to one of the format IDs above.

## Training History & Diagnostics

### 2026-05-05: Fix PPO Learning Pipeline

**Problem**: Agent stuck at ~45% win rate against `random_no_switch` after 5.5M steps. Policy and value loss stable (no learning).

**Root causes found**:
1. **Gradient clipping too aggressive**: `grad_clip=0.5` but total gradient norm was ~65. All gradients scaled to 0.77% — effectively zero learning rate for transformer/embedding layers. Fixed: `grad_clip=5.0`.
2. **No per-step action signal**: Removed both `matchup_reward_weight` and `action_quality_weight` to combat reward hacking, but this eliminated per-step credit assignment. Restored at low weights (0.2, 0.3) that don't dominate terminal ±10.
3. **Entropy collapse**: `entropy_coeff=0.005` gave negligible exploration bonus (~0.02 vs ~5.0 total loss). Fixed: `0.05`.
4. **Discount horizon too long**: `gamma=0.99` gives ~100-step horizon for ~25-turn battles. Fixed: `gamma=0.95` (~20-step horizon).
5. **Value function instability**: `vf_clip_param=25.0` allowed 3x the reward range per update. Fixed: `5.0`.
6. **LSTM crash**: `_extract_state` raised `ValueError` instead of returning zeros when state was missing. Fixed: return zeros.
7. **Self-play had no memory**: `SelfPlayPlayer` started each turn with zero LSTM state. Fixed: per-battle LSTM state cache.

**Config changes applied**:

| Parameter | Before | After |
|-----------|--------|-------|
| `grad_clip` | 0.5 | 5.0 |
| `gamma` | 0.99 | 0.95 |
| `entropy_coeff` | 0.005 | 0.05 |
| `vf_clip_param` | 25.0 | 5.0 |
| `hp_value_weight` | 1.0 | 2.0 |
| `matchup_reward_weight` | 5.0→0.0 | 0.2 |
| `action_quality_weight` | 2.0→0.0 | 0.3 |

**Curriculum**: Replaced self-play-first with 3-stage ramp: `random_warmup` (70% WR vs random_no_switch) → `self_play` (85% WR, 80/20 self/random) → `mixed` (terminal, 50/30/20 heuristic/self/random).

### 2026-05-06: Self-Play Diagnostics & NaN Fix

**Problem**: Self-play opponent still lost 90%+ after fixing weight loading. Needed instrumentation to diagnose.

**Root cause**: PyTorch 2.10 NaN bug in `TransformerEncoderLayer.forward()` with float additive `src_mask`. The model's learnable attention bias (`attn_bias = nn.Parameter(...)`) triggered the bug every forward pass, causing NaN logits → `multinomial` error → fallback to random.

**Fix**: Replaced `layer(x, src_mask=bias)` with manual norm-first decomposition (norm1 → self_attn → residual → norm2 → FFN → residual). See `_transformer_forward` in `battle_transformer.py`.

**Diagnostics added**:
- `SelfPlayPlayer._diag` accumulator: weight_load_count, fallback_count, action_histogram, top_prob, entropy, valid_action_count
- `pop_diagnostics()` → `CurriculumSingleAgentWrapper.pop_selfplay_diagnostics()` → `collect_selfplay_diagnostics(algo)` → trainer logs to `logs/selfplay_diagnostics.log` + MLflow
- `scripts/diagnose_selfplay.py`: entry point with 30% self-play, never promotes

**Training results** (500k steps, standard preset):
- `fallback_rate = 0%`, `mapping_fallback_rate = 0%` — self-play fully functional
- `avg_top_prob` 0.21 → 0.47 over training
- Win rates: 91% vs random, 31% vs random_no_switch, 30% vs self (10 games only)

### 2026-05-06: Expanded Validation System

**Problem**: Validation used subprocess per protocol (30-60s overhead), only argmax actions, missing `random_no_switch` opponent, no unified benchmark, no console output during training.

**Changes**:
- **Benchmark protocol**: 3 opponents × 50 episodes (150 total). Reports per-opponent WR with Wilson 95% CI, skill score (weighted: random=1.0, random_no_switch=1.5, heuristic=2.0), consistency (% tiers > 50% WR).
- **In-process validation**: `run_inprocess_validation()` runs against live algo during training — no subprocess, no Ray init, no checkpoint restore. Default during training.
- **Explore mode**: `--explore` flag samples from masked softmax instead of argmax, evaluating the stochastic policy.
- **Console output**: `format_validation_summary()` prints per-opponent win rate bars with CI bounds at each validation interval.
- **Config**: `ValidationScheduleConfig` defaults changed to `protocols=["benchmark"]`, new `benchmark_episodes_per_opponent` and `benchmark_opponents` fields.

**MLflow metrics logged**: `benchmark/skill_score`, `benchmark/consistency`, `benchmark/win_rate_vs_*`, `benchmark/ci_lower_vs_*`, `benchmark/ci_upper_vs_*`.

### 2026-05-06: Value Function Fix — Reward Scaling

**Problem**: Explained variance stuck at ~0.1 for entire project history. Value loss showed no downward trend.

**Root cause**: Returns in ~[-15, +15] range with no normalization. RLlib normalizes advantages but not returns, so the value function was chasing an unscaled, moving target.

**Fix**: Added `RewardConfig.reward_scale = 0.1` (applied in `battle_env.py`). Also lowered `vf_clip_param` 10.0→3.0, `lambda_` 0.92→0.88, widened model hidden_dim 256→512 / num_heads 4→8.

**Results** (378k steps):
- Explained variance: ~0.1 → **0.53–0.87** (median ~0.72)
- Value loss: high/no trend → **0.08–0.13** (converged by step 15k)
- Benchmark at 200k: 100% vs random, 96% vs random_no_switch, 22% vs heuristic

### 2026-08-17: Iteration profile + cheap self-play

**Profile** (`quick`, 12×4 envs, 8 Showdown servers, RTX 5090, 3 PPO iters, real ~48–53 turn battles). Script: `scripts/profile_training_iteration.py`. Raw: `logs/profile_training_iteration.json`.

| Piece | Result |
|--------|--------|
| Throughput | ~2100 env steps/s |
| `algo.train()` | ~2.0 s / iteration |
| Driver `torch.save` | ~2 ms |
| All `foreach_env` RPCs | ~6 ms |
| Ray worker CPU | ~590% (bottleneck) |
| Showdown CPU | ~50–70% after connect spike |
| GPU util | ~6–14% |

Driver housekeeping is not the wall on `quick`. Env-runner Python is. `standard` / `pure_league_play` may differ — re-profile before treating Showdown as the limit.

**`uv run` trap**: Ray workers inherit `uv run`, see no `.venv` in Ray's temp cwd, and download Torch on every worker. Launch Ray jobs with `.venv/bin/python`, not `uv run`.

**Self-play CPU waste**: `SelfPlayPlayer` used to `deepcopy` the snapshot and `load_state_dict` it **every opponent turn**. Inference is `eval()` + `no_grad()`, so the CPU copy is not mutated. Now loads once per episode. Trainer skips `selfplay_latest.pt` export when the active mix has no `self` / `historical` (e.g. league warmup). Exports again when promoting into a mix that needs it.

## Key Conventions

- **Package manager**: `uv` (not pip) for install/lint (`uv sync`, `uv run ruff`). Launch Ray training/validation with `.venv/bin/python`, not `uv run`.
- **Python version**: 3.13 (specified in `.python-version`)
- **Linting**: `ruff` (configured as dev dependency, no separate config file)
- **No formal test suite** — manual testing via `examples/training/test_training_run.py` and the `--preset quick` training run.
- **Multi-server architecture**: Training uses 8 parallel Pokémon Showdown instances (ports 8000-8007). Server scripts are in `scripts/`.
- **MLflow**: Training runs log to a local SQLite store (`sqlite:///mlflow.db`). View with `mlflow ui --backend-store-uri sqlite:///mlflow.db`. HTTP tracking URIs are ignored (the remote server is gone). The experiment name is `Pokemon_RL_Battler` / `Pokemon_RL_Marvin_Random` depending on config.
- **Checkpoints**: Saved to `checkpoints/` directory (gitignored). Use `--resume-checkpoint latest` to continue from the most recent.
