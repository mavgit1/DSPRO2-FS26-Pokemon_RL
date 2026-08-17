# Chapter 11: Distributed Infrastructure

## How Ray Makes Everything Parallel

This chapter explains how the training pipeline runs across dozens of processes
simultaneously. Understanding this is key to debugging performance issues.

---

## CONCEPT: Why Distributed Training?

A single battle environment produces one observation per turn. A Pokemon battle lasts ~20 turns. To collect 4096 steps for one PPO update:
- 1 environment: ~4096/20 = 205 sequential battles, ~3-5 minutes per update
- 192 environments: ~4096/20 = 205 parallel battles in ~2 seconds

The speedup is ~150x. This is why we use Ray.

---

## Ray Architecture

```
                                    +------------------+
                                    |  Driver Process  |
                                    |  (trainer.py)    |
                                    |  - Orchestrates  |
                                    |  - Updates model |
                                    +--------+---------+
                                             |
                                        Ray Core
                                    (shared memory, IPC)
                                             |
                    +------------------------+------------------------+
                    |                        |                        |
            +-------v-------+        +-------v-------+       +-------v-------+
            | Worker 0      |        | Worker 1      |  ...  | Worker 23     |
            | (CPU core)    |        | (CPU core)    |       | (CPU core)    |
            +-------+-------+        +-------+-------+       +-------+-------+
                    |                        |                        |
              +-----+-----+           +-----+-----+           +-----+-----+
              | Env 0     |           | Env 0     |           | Env 0     |
              | Env 1     |           | Env 1     |           | Env 1     |
              | ...       |           | ...       |           | ...       |
              | Env 7     |           | Env 7     |           | Env 7     |
              +-----+-----+           +-----+-----+           +-----+-----+
                    |                        |                        |
                 :8000                    :8001                    :8007
                    |                        |                        |
              +-----+-----+           +-----+-----+           +-----+-----+
              | Showdown 0 |          | Showdown 1 |    ...    | Showdown 7 |
              +------------+          +------------+           +------------+
```

### Roles

| Component | Process | What It Does |
|-----------|---------|-------------|
| Driver | Main process | Runs PPO updates on GPU, coordinates workers |
| Workers | Separate processes (24) | Run environments, collect experience |
| Environments | Within workers (8 each) | Play Pokemon battles on Showdown |
| Showdown | Separate Node.js processes (8) | Simulate battles |

---

## The Env Bridge Pattern

The driver needs to communicate with environments running on workers. This is the "env bridge":

```python
# env_bridge.py

def foreach_env(algo, fn):
    """Execute a function on every environment across all workers."""
    results = []
    for worker in algo.workers:
        for env in worker.envs:
            result = fn(env)
            results.append(result)
    return results
```

### What Flows Through the Bridge

| Direction | Data | When |
|-----------|------|------|
| Driver -> Workers | Curriculum stage update | When stage changes |
| Driver -> Workers | New opponent mix | When stage changes |
| Driver -> Workers | New reward config | When stage changes |
| Workers -> Driver | Battle outcomes (win/loss) | Every training iteration |
| Workers -> Driver | Episode statistics | Every training iteration |
| Workers -> Driver | Observation samples | For diagnostics |
| Workers -> Driver | Self-play diagnostics | Every training iteration |
| Workers -> Driver | Memory sentinels | For memory leak detection |

### Key Bridge Functions

```python
# Collect outcomes from all workers
outcomes = collect_recent_outcomes(algo)

# Push curriculum update to all workers
apply_curriculum_stage(algo, payload)

# Collect self-play diagnostics
diag = collect_selfplay_diagnostics(algo)

# Check memory usage
memory = collect_env_memory_sentinels(algo)
```

### The Wrapper Traversal Problem

RLlib wraps environments in multiple layers:

```
Worker
  -> VectorEnv
    -> MultiAgentEnvWrapper
      -> CurriculumSingleAgentWrapper  <-- We want to reach THIS
        -> PokemonBattleEnv
          -> poke-env SinglesEnv
```

The bridge must traverse this hierarchy to reach our custom wrapper:

```python
def _unwrap_env(env):
    """Navigate through RLlib's wrapper chain."""
    while hasattr(env, 'env'):
        env = env.env
    return env  # Eventually reaches CurriculumSingleAgentWrapper
```

---

## Resource Allocation

### For the `optimal` Preset

```
Total System Resources:
  GPU:  1x RTX 5090 (32GB VRAM)
  CPU:  24+ cores
  RAM:  64GB+

Allocation:
  Driver (GPU):  Model training, PPO updates
  Worker 0-23:   1 CPU core each, ~2GB RAM each
    Env 0-7:     ~256MB each (Showdown connection + battle state)

  Showdown 0-7:  Node.js, ~100MB each
```

### GPU vs CPU Split

- **GPU**: Only used for PPO updates (forward + backward pass on the model)
- **CPU**: Everything else (environment rollout, Showdown communication, metric collection)

The model runs inference on CPU during rollout (collecting experience) and on GPU during training (updating weights). This is standard for RLlib.

---

## Checkpoint Flow

```
Training Step N
    |
    v
[Save checkpoint to checkpoints/step_XXXXXX/]
    | - Model weights
    | - Optimizer state
    | - Training step counter
    |
    v
[Export self-play weights to checkpoints/selfplay_latest.pt]
    | - Model weights only (no optimizer)
    | - Overwritten every iteration
    |
    v
Workers detect new weights file via mtime check
    |
    v
Self-play opponents reload weights
```

### Checkpoint Contents

```
checkpoints/step_200000/
  |-- algorithm_state.pkl     # Full algorithm state (can restore training)
  |-- policy_state.pkl        # Policy weights and optimizer
  |-- replay_buffer.pkl       # Any buffered experience
  |-- rllib_state.json        # Training metadata
```

### Resume Flow

```bash
uv run train_battler.py --preset optimal --resume-checkpoint latest --mlflow-run-id abc123
```

1. `latest` resolves to the most recent checkpoint directory
2. `algo.restore(checkpoint_path)` loads all state
3. Training continues from step 200,001
4. MLflow logs to the same run (no duplicate runs)

---

## Server Management

### Spinning Up Servers

```bash
./scripts/spin_up_multiple_showdown.sh
# Starts 8 Pokemon Showdown instances on ports 8000-8007
```

Each server is a separate Node.js process running the Pokemon Showdown battle simulator.

### Custom Formats

```bash
./scripts/setup_custom_formats.sh
# Creates pokemon-showdown/config/custom-formats.ts
# Adds no-gimmick format variants
```

Must be run before spinning up servers. Adds formats like `gen8randombattlenogimmicks` that disable Dynamax and Sleep Clause.

### Tearing Down

```bash
./scripts/kill_all_showdown.sh
# Kills all Showdown processes
```

---

## Common Infrastructure Issues

### Workers Can't Find Files

Ray workers run from `/tmp/ray/...`. Any relative path will fail:
```python
# BAD: Relative path
path = "checkpoints/selfplay_latest.pt"

# GOOD: Absolute path
path = os.path.abspath("checkpoints/selfplay_latest.pt")
```

### Port Exhaustion

With 8 servers on ports 8000-8007 and 192 environments, each server handles ~24 concurrent connections. If Showdown is slow, connections queue up and environments time out.

### Memory Leaks

Each environment holds battle state, poke-env connections, and Showdown data. With 192 environments, even small leaks compound:
```python
# Memory sentinel tracking
memory = collect_env_memory_sentinels(algo)
# Reports: number of active battles, cached states, etc.
```

### Dead Workers

If a worker crashes (Showdown timeout, OOM), Ray restarts it. But the restarted worker loses its environment state and opponent pool. The env bridge handles this gracefully by skipping workers that don't respond.

---

## The Full Data Pipeline

```
Showdown sends JSON events
         |
         v
poke-env parses into Python objects
         |
         v
BattleEnv computes obs (13, 164) + reward + action_mask
         |
         v
Ray Worker collects obs from 8 envs
         |
         v
Ray sends batch of observations to Driver
         |
         v
Driver runs model forward on GPU (policy + value)
         |
         v
Driver computes PPO loss, updates model
         |
         v
Driver exports updated weights for self-play
         |
         v
Next iteration begins
```

Round-trip time: ~35-70 seconds per iteration (4096 steps).

---

## What's Next

Chapter 12 covers the full history of reward engineering - the most iterative and debug-heavy part of the project.
