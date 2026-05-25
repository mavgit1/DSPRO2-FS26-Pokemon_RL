# Debug Tools

Observation pipeline validation and local MLflow for diagnosing training issues.

## Quick Start

```bash
# 1. Start Showdown servers
./scripts/spin_up_multiple_showdown.sh

# 2. Run standalone pipeline validator (5 battles, no training)
uv run scripts/debug/validate_pipeline.py

# 3. Optionally start local MLflow to inspect results
./scripts/debug/start_local_mlflow.sh
```

## Tools

### `validate_pipeline.py` — Standalone Pipeline Check

Runs real poke-env random battles and validates every observation. Checks:

| Check | What it catches |
|---|---|
| Shape correctness | Wrong obs/mask dimensions |
| NaN detection | NaN values in observation arrays |
| Active token layout | Token 1 = active, bench tokens not active |
| Weather encoding | Weather dims always zero (Bug #1 symptom) |
| Action mask binary | Non-binary mask values |
| Switch ↔ fainted consistency | Fainted bench mon not masked out (Bug #2 symptom) |
| Bench fainted flags | Fainted mons dropped from tokens instead of kept |
| Species IDs | Active tokens missing species |

```bash
uv run scripts/debug/validate_pipeline.py --port 8000 --num-battles 10
```

### `run_debug_training.py` — Short Training with Validation

50k-step training run (tiny model) that runs the observation validator every iteration. Logs to local MLflow.

```bash
# Start local MLflow first
./scripts/debug/start_local_mlflow.sh

# In another terminal:
uv run scripts/debug/run_debug_training.py
```

Open `http://localhost:5000` → experiment `debug_observation_pipeline` to see `obs_val/*` metrics alongside training metrics.

Environment variables:
- `MLFLOW_TRACKING_URI` — defaults to `http://localhost:5000`
- `OBS_VAL_FREQ` — validate every N iterations (default: 1)
- `NUM_SERVERS` — number of Showdown servers (default: 8)
- `START_PORT` — starting port (default: 8000)

### `start_local_mlflow.sh` — Local MLflow Server

```bash
./scripts/debug/start_local_mlflow.sh          # port 5000
./scripts/debug/start_local_mlflow.sh 5001     # custom port
```

Data stored in `.mlflow/local_tracking/` (gitignored).

## Architecture

```
src/debug/
  observation_validator.py   # Reusable validation module (importable from trainer)

scripts/debug/
  validate_pipeline.py       # Standalone: run battles, validate observations
  run_debug_training.py      # Training run with validation hooks
  start_local_mlflow.sh      # Local MLflow server
```

The validator is designed to be called from anywhere — training loop, callbacks, or standalone scripts. It takes a list of observation dicts and returns flat metrics suitable for MLflow:

```python
from src.debug.observation_validator import validate_observations
metrics = validate_observations(obs_samples)
mlflow.log_metrics(metrics, step=global_step)
```

## Bugs Fixed

These tools were created to verify fixes for two critical bugs:

1. **Weather never encoded** (`embedding.py:290-294`) — `battle.weather` is `Dict[Weather, int]`, code passed the whole dict to `_get_list_index()` instead of iterating keys. Fixed to `for weather in battle.weather:`.

2. **Switch token ↔ action mismatch** (`action_space.py`) — Observation included fainted pokemon in bench tokens but compressed switch actions used dense legal-only indexing. Fixed with bench-relative indexing so compressed switch 8+k always maps to bench token 2+k.
