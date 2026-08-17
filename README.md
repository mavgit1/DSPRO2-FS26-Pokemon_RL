# DSPRO2: Pokemon RL Battler

Reinforcement learning project training PPO agents to play Pokemon battles. Uses Ray RLlib, PyTorch, `poke-env` against local Pokemon Showdown servers, and MLflow for experiment tracking.

## Setup

### Prerequisites

- **Node.js** (Showdown server) — via `nvm`
- **Python 3.13** — via `uv`

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

### 3. Environment / MLflow

Tracking is local. No MLflow server or credentials are required.

```bash
cp .env-example .env
# Optional: RAY_TMPDIR=/path/to/ray/tmp
```

`.env-example` sets `MLFLOW_TRACKING_URI=sqlite:///mlflow.db`. HTTP tracking URIs (the old remote server) are ignored and rewritten to this local SQLite store.

- Runs land in `mlflow.db` (gitignored). Artifacts go under `mlruns/`.
- `--mlflow-run-id` from the old remote will not resolve. Checkpoint resume (`--resume-checkpoint`) still works; MLflow history starts fresh.
- Training prints the tracking URI and run id at start.

View the UI:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

### 4. Pokemon Showdown

```bash
git clone https://github.com/smogon/pokemon-showdown.git
cd pokemon-showdown && npm install && cp config/config-example.js config/config.js && cd ..
./scripts/setup_custom_formats.sh
```

## Running

### Showdown servers

Training (and checkpoint validation) checks whether the configured Showdown ports are reachable and starts any that are missing. `pokemon-showdown` must already be cloned and installed as above. You can still manage servers by hand:

```bash
./scripts/setup_training.sh          # first time (starts 8 servers on 8000–8007)
./scripts/spin_up_multiple_showdown.sh
./scripts/kill_all_showdown.sh
```

### Start training with the venv Python, not `uv run`

`uv sync` is the right way to **install**. Do **not** start Ray training with `uv run train_battler.py`.

Ray workers run from a temporary working directory. If the driver was launched via `uv run`, those workers inherit `uv run` as well. `uv` then sees that the temp dir has no matching `.venv`, creates a **new** environment, and starts downloading Torch (~900 MB) on every worker. Training looks hung at "Creating virtual environment" / "Downloading torch".

Use the project interpreter directly:

```bash
.venv/bin/python train_battler.py --preset quick
```

Or activate first:

```bash
source .venv/bin/activate
python train_battler.py --preset quick
```

Same rule for anything that starts Ray (validation, self-play diagnostics, hyperparameter sweeps, `scripts/profile_training_iteration.py`). `uv run ruff` for lint is fine.

### Iteration profile (`quick`, 2026-08-17)

Timed 3 PPO iterations on this machine (12 workers × 4 envs, 8 Showdown servers, RTX 5090) with `scripts/profile_training_iteration.py`. Battles were real (~48–53 turns). Raw numbers: `logs/profile_training_iteration.json`.

| Piece | Result |
|--------|--------|
| Throughput | ~2100 env steps/s |
| `algo.train()` | ~2.0 s / iteration (almost the whole loop) |
| Self-play `torch.save` on the driver | ~2 ms |
| All `foreach_env` metric RPCs | ~6 ms |
| Ray worker CPU | ~590% (the bottleneck) |
| Showdown CPU | ~50–70% after the first connection spike (not saturated) |
| GPU util | ~6–14% (learner idle) |

So on `quick`, driver housekeeping is not the wall. Env-runner Python (embed + policy sample + self-play inference) is. `standard` / `pure_league_play` may look different (heavier net, more heuristic). Re-run the profiler with those presets before treating Showdown as the limit.

Self-play used to `deepcopy` + `load_state_dict` the frozen snapshot **every opponent turn**. Inference is `eval()` + `no_grad()`, so that reload was wasted worker CPU. The opponent now loads once per episode. The driver also skips the `.pt` export when the active mix has no `self` / `historical`.

```bash
.venv/bin/python scripts/profile_training_iteration.py --preset quick --iterations 3 --num-servers 8
```

### Training presets

| Preset | Teams | Notes |
|--------|--------|--------|
| `quick`, `standard`, `optimal`, `memory_safe`, `large` | **No** — default `gen8randombattlenogimmicks`, Showdown random teams | General curriculum / resource profiles |
| `pure_league_play` | **Yes** — 3→5→10→20 team pool, `gen8customgame` | Production curriculum + league mix (25M default) |
| `pure_league_pool` | Same as `pure_league_play` | Alias only |

Only **`pure_league_play`** uses `team_pool_manifest` and per-stage pool sizes. All other presets leave `player_team_path` and `team_pool_manifest` unset, so the agent plays **random battles** with random teams.

```bash
.venv/bin/python train_battler.py --preset quick
.venv/bin/python train_battler.py --preset pure_league_play --num-servers 8 --timesteps 25000000
```

Pool manifests live under `data/teams/pool_curriculum/` (regenerate with `python scripts/build_team_pool_manifest.py`).

**Resume** (e.g. continue in `league_training` after a crash):

```bash
RESUME_CURRICULUM_STAGE=league_training MLFLOW_RUN_ID=<run_id> \
  .venv/bin/python train_battler.py --preset pure_league_play \
  --resume-checkpoint checkpoints/step_XXXXXX --timesteps 25000000
```

### Validation

```bash
.venv/bin/python scripts/validate_checkpoint.py \
  --checkpoint checkpoints/step_XXXXXX \
  --protocol benchmark \
  --preset pure_league_play
```

Scheduled training validation uses the **`benchmark`** protocol (random battle format, random agent teams). That measures out-of-distribution play vs the custom team-pool league setup. For in-distribution eval, use `fixed_paired` / `mirror` with the validation manifests or extend benchmark to pass the team pool.

### Diagnostics

```bash
.venv/bin/python scripts/analyze_model_diagnostics.py
.venv/bin/python scripts/diagnose_selfplay.py --preset standard --timesteps 500000
.venv/bin/python scripts/hparam_sweep.py --n-trials 3 --timesteps 100000
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
| `profile_training_iteration.py` | Time rollout vs driver RPCs vs CPU/GPU |
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
