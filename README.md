# DSPRO2: Pokemon RL Battler

Reinforcement learning project training PPO agents to play Pokemon battles (BDSP/Nuzlocke). Uses Ray RLlib for distributed training, PyTorch for neural networks, `poke-env` as the Gymnasium-compatible interface to local Pokemon Showdown servers, and MLflow for experiment tracking.

## Setup

### Prerequisites

- **Node.js** (for the Pokemon Showdown server) — managed via `nvm`
- **Python 3.13** — managed via `uv`
- **MLflow credentials** — ask a team member

### 1. Install `nvm` and Node.js

```bash
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.4/install.sh | bash
# Restart your terminal, then:
nvm install 22.12.0
nvm use 22.12.0
```

### 2. Install `uv` and Python Dependencies

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync
```

This creates a local `.venv` and installs the exact versions from `uv.lock`.

### 3. Environment Variables and MLflow

```bash
cp .env-example .env
```

Edit `.env` and fill in the MLflow credentials:

```ini
MLFLOW_TRACKING_URI="https://mlflow-server-url.com"
MLFLOW_TRACKING_USERNAME="Username"
MLFLOW_TRACKING_PASSWORD="Password"
```

### 4. Pokemon Showdown Server

```bash
git clone https://github.com/smogon/pokemon-showdown.git
cd pokemon-showdown
npm install
cp config/config-example.js config/config.js
cd ..
```

If you encounter server throttling during training, adjust rate limits in `pokemon-showdown/config/config.js`.

### 5. Custom Showdown Formats

Sets up no-gimmick battle formats (no Dynamax, no Terastallize, no Sleep Clause). Run once, then restart servers.

```bash
./scripts/setup_custom_formats.sh
```

## Running

### Start Servers

```bash
# First-time setup (also starts 8 servers on ports 8000-8007)
./scripts/setup_training.sh

# Start servers (after initial setup)
./scripts/spin_up_multiple_showdown.sh

# Stop all servers
./scripts/kill_all_showdown.sh
```

### Training

```bash
uv run train_battler.py --preset quick              # quick test run
uv run train_battler.py --preset standard             # default
uv run train_battler.py --preset optimal              # RTX 5090
uv run train_battler.py --preset memory_safe          # reduced RAM
uv run train_battler.py --preset large                # max resources
uv run train_battler.py --preset pure_league_play     # curriculum + league mix (recommended)
```

Presets are defined in `src/config/TM_optimal_config.py`. See `mav_README.md` for the full `pure_league_play` curriculum, reward, and diagnostics workflow.

**Recommended production-style run (1.5M steps):**

```bash
./scripts/setup_training.sh 8
set -a && [ -f .env ] && . ./.env && set +a
PYTHONUNBUFFERED=1 SAVE_LEAGUE_HISTORY=1 uv run --active train_battler.py \
  --preset pure_league_play --num-servers 8 --timesteps 1500000
```

The policy uses poke-env's **native 22-action** space (switches 0–5, moves 6–9, gimmicks 10–21). Training writes decision diagnostics to `logs/validation/decision_diagnostics_samples.json` for post-hoc analysis.

### Resume Training

Resume from a checkpoint and continue logging into the same MLflow run:

```bash
uv run train_battler.py --preset optimal \
  --resume-checkpoint latest \
  --mlflow-run-id <RUN_ID>
```

- `--resume-checkpoint latest` picks the newest `checkpoints/step_<n>/` directory
- Pass a specific path instead of `latest` to resume from a particular checkpoint
- `--resume-checkpoint` alone resumes model state but creates a new MLflow run
- `--mlflow-run-id` alone continues logging but starts from a fresh model

### Post-training forensics

After a run, analyze what the policy attended to and how confident it became:

```bash
uv run python scripts/analyze_model_diagnostics.py
```

This reads `logs/validation/decision_diagnostics_samples.json` (collected during training) and writes plots under `logs/validation/diagnostics_plots/` plus `diagnostics_report.json` and `diagnostics_report.md`. For win rates against fixed opponents, use validation below.

### Validation

Benchmark a checkpoint against 3 opponent tiers (random, random_no_switch, heuristic):

```bash
uv run scripts/validate_checkpoint.py \
  --checkpoint checkpoints/step_XXXXXX \
  --protocol benchmark \
  --preset standard

# Stochastic policy (masked softmax sampling instead of argmax)
uv run scripts/validate_checkpoint.py \
  --checkpoint checkpoints/step_XXXXXX \
  --protocol benchmark \
  --preset standard \
  --explore

# Quick 3-episode smoke test
uv run scripts/validate_checkpoint.py \
  --checkpoint checkpoints/step_XXXXXX \
  --protocol smoke \
  --preset quick
```

### Self-Play Diagnostics

30% self-play run that never promotes from stage 0. Useful for diagnosing opponent quality.

```bash
uv run scripts/diagnose_selfplay.py --preset standard --timesteps 500000
```

### Hyperparameter Sweep

Optuna TPE sweep, 500k steps per trial. Resumes from a SQLite database.

```bash
uv run scripts/hparam_sweep.py --n-trials 50                # full sweep (~14h)
uv run scripts/hparam_sweep.py --n-trials 3 --timesteps 100000  # dry run
```

### Linting and Formatting

```bash
uv run ruff check .
uv run ruff format .
```

### Dependency Management

```bash
uv add <package>        # add a dependency
uv remove <package>     # remove a dependency
uv cache clean          # clean cache periodically
```

Always commit `uv.lock` and `pyproject.toml` after dependency changes.

## Project Structure

```
train_battler.py          Entry point for training
src/
  config/                 Training, hardware, and reward configurations
  envs/                   Gymnasium environments wrapping Pokemon Showdown
  models/                 Neural network architectures (battle_transformer.py)
  teams/                  Pokemon team generation
  training/               Training orchestration
    trainer.py            PokemonTrainer — wires the full training lifecycle
    rllib_config_builder.py   PPO config and environment registration
    env_bridge.py         Worker-side bridge for curriculum and metrics
    callbacks.py          Curriculum progression and checkpoint management
    curriculum.py         Progressive difficulty scaling
    resume.py             Checkpoint path resolution
    metrics/              Metric extraction and aggregation
    monitoring/           Runtime telemetry (CPU/RAM/GPU)
  data/                   Dataset utilities
  validation/             Checkpoint evaluation and benchmarking
scripts/                  Server management, diagnostics, sweeps
  analyze_model_diagnostics.py   Plots + AI-readable reports from training samples
  visualize_decision_diagnostics.py   Plot-only wrapper
src/analysis/             Shared decision diagnostics logic
data/                     BDSP trainer CSVs, team manifests, gauntlet order
examples/                 Sandbox scripts, notebooks, reference players
```
