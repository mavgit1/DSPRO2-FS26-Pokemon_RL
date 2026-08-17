# Chapter 01: The Entry Point

## What Happens When You Run `uv run train_battler.py --preset optimal`

This chapter traces exactly what happens from the moment you press Enter to the moment training begins.

---

## The Command

```bash
uv run train_battler.py --preset optimal
```

`uv` is the package manager (like pip, but faster, with lockfiles). `uv run` executes the script in the project's virtual environment with all dependencies available.

---

## Step 1: Environment Setup

```python
from dotenv import load_dotenv
load_dotenv()
```

This loads `.env` which contains MLflow credentials (tracking URI, username, password). Without this, the trainer can't log experiments. The `.env` file is gitignored - it should never be committed.

---

## Step 2: Argument Parsing

```python
parser = argparse.ArgumentParser()
parser.add_argument("--preset", default="standard",
                    choices=["quick", "standard", "memory_safe", "optimal", "large"])
parser.add_argument("--timesteps", type=int)
parser.add_argument("--num-servers", type=int, default=8)
parser.add_argument("--start-port", type=int, default=8000)
parser.add_argument("--debug", action="store_true")
parser.add_argument("--resume-checkpoint", type=str)
parser.add_argument("--mlflow-run-id", type=str)
# ... more args
args = parser.parse_args()
```

### Key Arguments Explained

| Argument | What It Does | When You'd Use It |
|----------|-------------|-------------------|
| `--preset` | Selects a hardware/config preset | Always. `optimal` for RTX 5090, `quick` for testing |
| `--timesteps` | Overrides total training steps | Quick experiments |
| `--num-servers` | How many Showdown instances | Must match running servers |
| `--start-port` | First port for Showdown | Default 8000, 8 servers = ports 8000-8007 |
| `--resume-checkpoint` | Path to checkpoint to resume from | After interruption. `latest` auto-resolves |
| `--mlflow-run-id` | Continue logging to an existing MLflow run | Resume without creating duplicate runs |
| `--disable-scheduled-validation` | Turn off automatic validation | Faster training, no evaluation overhead |

---

## Step 3: Configuration Loading

```python
from src.config.TM_optimal_config import get_config, resolve_mlflow_experiment_for_training

config = get_config(args.preset)
```

> See **Chapter 02** for the full config deep dive.

`get_config("optimal")` returns a `TrainingConfig` dataclass containing every parameter the project needs: model architecture, PPO hyperparameters, environment settings, curriculum stages, and validation schedule.

---

## Step 4: MLflow Experiment Setup

```python
mlflow.set_experiment(
    resolve_mlflow_experiment_for_training(config, args.resume_checkpoint)
)
```

This creates or re-opens an MLflow experiment. The experiment name is typically `"Pokemon_RL_Battler"`. If resuming, it re-opens the same experiment to keep runs grouped.

---

## Step 5: Trainer Creation

```python
from src.training.trainer import PokemonTrainer

trainer = PokemonTrainer(
    config=config,
    preset=args.preset,
    num_servers=args.num_servers,
    start_port=args.start_port,
    resume_checkpoint=args.resume_checkpoint,
    mlflow_run_id=args.mlflow_run_id,
)
```

The `PokemonTrainer` constructor:
1. Stores the config and preset name
2. Sets up the curriculum manager (if curriculum is configured)
3. Initializes a rolling window for win rate tracking
4. Creates a checkpoint manager
5. Prepares diagnostics collectors

At this point, nothing heavy has happened yet. No Ray, no model, no environments.

---

## Step 6: Training Starts

```python
trainer.train()
```

This is the big one. `train()` handles:
1. Initializing Ray (distributed computing framework)
2. Registering the custom Gymnasium environment
3. Building the PPO algorithm with the transformer model
4. Optionally restoring from a checkpoint
5. Starting the MLflow run
6. Running the training loop (millions of steps)
7. Saving checkpoints, running validation, updating curriculum
8. Shutting down Ray when done

> See **Chapter 03** for the full training loop walkthrough.

---

## Complete Flow Diagram

```
uv run train_battler.py --preset optimal
         |
         v
    load_dotenv()          # Load MLflow credentials
         |
         v
    argparse.parse_args()  # Parse CLI arguments
         |
         v
    get_config("optimal")  # Build TrainingConfig dataclass
         |
         v
    mlflow.set_experiment()  # Connect to MLflow server
         |
         v
    PokemonTrainer(config, ...)  # Create trainer (lightweight init)
         |
         v
    trainer.train()        # THE BIG ONE (Chapter 03)
         |
         +---> ray.init()           # Start distributed runtime
         +---> register_environments()  # Tell Ray about our env
         +---> build_ppo_config()   # Configure PPO algorithm
         +---> PPO(config)          # Create algorithm + model
         +---> [restore checkpoint] # If resuming
         +---> TRAINING LOOP:
         |      +---> algo.train()  # Collect experience, update model
         |      +---> export self-play weights
         |      +---> update curriculum stage
         |      +---> run validation
         |      +---> save checkpoint
         |      +---> log metrics to MLflow
         +---> ray.shutdown()       # Clean up
```

---

## What Each Dependency Does

You'll see these imports throughout the project. Here's what each one is for:

| Library | Role in This Project |
|---------|---------------------|
| `ray` | Distributed computing. Runs environments on separate workers, collects experience in parallel. |
| `rllib` | RL library built on Ray. Provides the PPO algorithm, training loop, checkpointing. |
| `torch` | PyTorch. The neural network framework. Our transformer model is built with it. |
| `poke-env` | Gymnasium interface to Pokemon Showdown. Turns battles into RL environment steps. |
| `mlflow` | Experiment tracking. Logs metrics, parameters, and artifacts for every run. |
| `numpy` | Array operations. Used for observations, metrics, and data processing. |
| `gymnasium` | The standard RL environment API. Our env implements the Gymnasium interface. |

---

## The Convenience Function

At the bottom of `train_battler.py`, there's also a `train()` function:

```python
def train(preset="standard", timesteps=None, ...):
    """Convenience function for quick training calls."""
```

This is a simpler interface that skips argparse entirely. Useful for notebooks or quick scripts where you don't need CLI flexibility.

---

## What's Next

Now you know the bootstrapping sequence. The next chapter dives into the configuration system - what all those parameters actually mean and how they're organized.
