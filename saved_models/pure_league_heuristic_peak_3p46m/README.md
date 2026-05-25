# pure_league_heuristic_peak_3p46m

Best heuristic benchmark snapshot from MLflow run `b86a99a5fb8343cbbeee419aff008ee8`.

| Field | Value |
|-------|--------|
| RLlib checkpoint | `checkpoint/` (copy of `step_3434222`) |
| Env steps at save | ~3,434,222 |
| Benchmark at step 3,458,882 | **30%** vs heuristic (50 eps, argmax) |
| Also | 96% random, 70% random_no_switch, 56% self |
| Preset | `pure_league_play` |
| Format | `gen8randombattlenogimmicks` |
| Training date | 2026-05-22 |

## Load for eval

```bash
uv run python scripts/validate_checkpoint.py \
  --checkpoint saved_models/pure_league_heuristic_peak_3p46m/checkpoint \
  --protocol benchmark \
  --preset pure_league_play
```

## Resume training (same architecture)

```bash
uv run train_battler.py --preset pure_league_play \
  --resume-checkpoint saved_models/pure_league_heuristic_peak_3p46m/checkpoint \
  --timesteps 4000000 \
  --mlflow-run-id b86a99a5fb8343cbbeee419aff008ee8
```
