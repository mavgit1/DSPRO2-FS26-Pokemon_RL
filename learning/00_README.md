# Learning Guide: Pokemon RL Battler

This folder is a comprehensive, top-down walkthrough of the entire project.
Read the chapters in order - each one follows the execution path of
`uv run train_battler.py --preset optimal`.

## Reading Order

| # | File | What You'll Learn |
|---|------|-------------------|
| 01 | `01_entry_point.md` | What happens when you press Enter. The CLI, argparse, and how everything bootstraps. |
| 02 | `02_config_deep_dive.md` | How `--preset optimal` becomes a full configuration object. Every dataclass, every parameter. |
| 03 | `03_training_loop.md` | Inside `PokemonTrainer.train()`. The main loop, step-by-step. |
| 04 | `04_environment.md` | How a Pokemon battle on Showdown becomes a Gymnasium environment. The poke-env bridge. |
| 05 | `05_observation_pipeline.md` | How a battle state becomes observation tensors: 13 tokens × 164 dense features + categorical embeddings (species/items/abilities). |
| 06 | `06_model_architecture.md` | The BattleTransformer: embeddings, attention bias, LSTM memory, policy/value heads. |
| 07 | `07_ppo_explained.md` | PPO theory from scratch, then how each hyperparameter maps to our code. |
| 08 | `08_curriculum_learning.md` | How the agent ramps from random opponents to self-play. Stage transitions, win rate gates. |
| 09 | `09_self_play.md` | How the agent plays against a copy of itself. Weight export, diagnostics, the NaN bug. |
| 10 | `10_validation.md` | How we measure agent quality. Benchmark protocol, Wilson CIs, skill scores. |
| 11 | `11_distributed_infrastructure.md` | How Ray distributes training across workers. The env bridge pattern. Multi-server architecture. |
| 12 | `12_reward_engineering.md` | The full history of reward shaping. What went wrong, what worked, and why. |
| 13 | `13_key_decisions.md` | Every major design decision in one place, with alternatives considered. |

## How to Use These Notes

1. **Read sequentially first.** Chapters build on each other. Chapter 4 won't make sense without chapter 3.
2. **Have the code open.** Every chapter references specific files and line numbers. Follow along.
3. **The boxes mean things:**
   - `CONCEPT` boxes explain the underlying theory (what is PPO, what is attention)
   - `IN OUR CODE` boxes show exactly where and how we implement something
   - `WHY` boxes explain design decisions and what went wrong before
   - `DIAGRAM` boxes have ASCII art of data flow
4. **Don't skip the WHY boxes.** This project has a lot of non-obvious decisions that came from debugging sessions. The "why" is often more important than the "what."

## Prerequisites

You should be vaguely familiar with:
- Python (classes, decorators, async/await basics)
- Basic ML concepts (neural networks, training loops, loss functions)
- The command line

Everything else (RL theory, transformers, Ray, poke-env) is explained from scratch in the relevant chapters.

## File Map

Quick reference for where to find things in the actual codebase:

```
train_battler.py                          # Entry point
src/config/TM_optimal_config.py           # All configuration
src/training/
  trainer.py                              # Main orchestrator
  rllib_config_builder.py                 # RLlib PPO config construction
  env_bridge.py                           # Worker communication layer
  callbacks.py                            # RLlib callback imports
  curriculum.py                           # Curriculum stage management
  resume.py                               # Checkpoint resolution
  checkpointing.py                        # Checkpoint saving
  metrics/                                # Metric extraction and aggregation
  monitoring/                             # System telemetry
src/models/
  battle_transformer.py                   # Transformer + LSTM model
  embedding.py                            # Observation -> tensor conversion
src/envs/
  battle_env.py                           # Gymnasium environment + wrapper
  random_no_switch_player.py              # Custom opponent
src/validation/
  runner.py                               # Validation execution
  protocols.py                            # Protocol definitions
  metrics.py                              # Metric computation
  reporting.py                            # MLflow + console output
  teams.py                                # Team generation
scripts/
  spin_up_multiple_showdown.sh            # Server management
  validate_checkpoint.py                  # Standalone validation entry point
  diagnose_selfplay.py                    # Self-play diagnostics
  hparam_sweep.py                         # Optuna hyperparameter sweep
```
