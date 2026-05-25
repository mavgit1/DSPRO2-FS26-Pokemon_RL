# DSPRO2: Pokemon RL Battler

Reinforcement learning project training PPO agents to play Pokemon battles. Uses Ray RLlib, PyTorch, `poke-env` against local Pokemon Showdown servers, and MLflow for experiment tracking.

## Setup

### Prerequisites

- **Node.js** (Showdown server) — via `nvm`
- **Python 3.13** — via `uv`
- **MLflow credentials** — ask a team member

### 1. Node.js

```bash
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.4/install.sh | bash
nvm install 22.12.0
nvm use 22.12.0
```

### 2. Python

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync
```

### 3. Environment

```bash
cp .env-example .env
# Set MLFLOW_TRACKING_URI, MLFLOW_TRACKING_USERNAME, MLFLOW_TRACKING_PASSWORD
```

### 4. Pokemon Showdown

```bash
git clone https://github.com/smogon/pokemon-showdown.git
cd pokemon-showdown && npm install && cp config/config-example.js config/config.js && cd ..
./scripts/setup_custom_formats.sh
```

## Running

### Showdown servers

```bash
./scripts/setup_training.sh          # first time (starts 8 servers on 8000–8007)
./scripts/spin_up_multiple_showdown.sh
./scripts/kill_all_showdown.sh
```

### Training presets

| Preset | Teams | Notes |
|--------|--------|--------|
| `quick`, `standard`, `optimal`, `memory_safe`, `large` | **No** — default `gen8randombattlenogimmicks`, Showdown random teams | General curriculum / resource profiles |
| `pure_league_play` | **Yes** — 3→5→10→20 team pool, `gen8customgame` | Production curriculum + league mix (25M default) |
| `pure_league_pool` | Same as `pure_league_play` | Alias only |

Only **`pure_league_play`** uses `team_pool_manifest` and per-stage pool sizes. All other presets leave `player_team_path` and `team_pool_manifest` unset, so the agent plays **random battles** with random teams.

```bash
uv run train_battler.py --preset quick
uv run train_battler.py --preset pure_league_play --num-servers 8 --timesteps 25000000
```

Pool manifests live under `data/teams/pool_curriculum/` (regenerate with `python scripts/build_team_pool_manifest.py`).

**Resume** (e.g. continue in `league_training` after a crash):

```bash
RESUME_CURRICULUM_STAGE=league_training MLFLOW_RUN_ID=<run_id> \
  uv run train_battler.py --preset pure_league_play \
  --resume-checkpoint checkpoints/step_XXXXXX --timesteps 25000000
```

### Validation

```bash
uv run scripts/validate_checkpoint.py \
  --checkpoint checkpoints/step_XXXXXX \
  --protocol benchmark \
  --preset pure_league_play
```

Scheduled training validation uses the **`benchmark`** protocol (random battle format, random agent teams). That measures out-of-distribution play vs the custom team-pool league setup. For in-distribution eval, use `fixed_paired` / `mirror` with the validation manifests or extend benchmark to pass the team pool.

### Diagnostics

```bash
uv run python scripts/analyze_model_diagnostics.py
uv run scripts/diagnose_selfplay.py --preset standard --timesteps 500000
uv run scripts/hparam_sweep.py --n-trials 3 --timesteps 100000
```

### Lint

```bash
uv run ruff check .
uv run ruff format .
```

## Scripts

**Core (keep):**

| Script | Purpose |
|--------|---------|
| `setup_training.sh`, `spin_up_multiple_showdown.sh`, `kill_all_showdown.sh` | Showdown servers |
| `setup_custom_formats.sh` | Custom battle formats |
| `validate_checkpoint.py` | Checkpoint eval |
| `build_team_pool_manifest.py` | Regenerate pool team JSONs |

**Useful if you need them:**

| Script | Purpose |
|--------|---------|
| `analyze_model_diagnostics.py`, `visualize_decision_diagnostics.py` | Training decision plots |
| `collect_checkpoint_diagnostics.py` | Checkpoint analysis export |
| `generate_embedding_vocab.py`, `audit_embedding_vocab.py` | Vocab rebuild / audit |
| `hparam_sweep.py`, `diagnose_selfplay.py` | Sweeps / self-play debugging |
| `generate_validation_team_manifest.py` | Build validation manifests |

## Project structure

```
train_battler.py              Training entry point
src/config/TM_optimal_config.py   Presets and curriculum
src/envs/                     Showdown gym env
src/models/                   Policy (battle_transformer)
src/training/                 Trainer, curriculum, self-play
src/validation/               Benchmark and paired-team eval
data/teams/pool_curriculum/   Staged team pools for pure_league_play
saved_models/                 Committed reference checkpoints
scripts/                      Ops and eval helpers
```
