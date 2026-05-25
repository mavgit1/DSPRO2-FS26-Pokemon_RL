# pure_league_play_25M

Final checkpoint from the pool-curriculum + league training run (25M steps).

| Field | Value |
|-------|--------|
| RLlib checkpoint | `checkpoint/` (copy of `checkpoints/final` on vast) |
| Self-play weights | `selfplay_latest.pt` (local copy; `*.pt` gitignored) |
| Env steps | 25,001,700 |
| Preset | `pure_league_play` |
| Curriculum | warmup → bridge → pool 3/5/10/20 @ 65% → `league_training` |
| League mix (final) | 68% heuristic, 5% historical, 10% self, 12% random, 5% rnd-no-switch |
| MLflow run | `331f74404401435fbada77344ef3b8e5` |
| Training date | 2026-05-25 |

Last scheduled benchmark (random-battle protocol, OOD vs train): ~14% vs heuristic @ step ~25M.

## Eval

```bash
uv run python scripts/validate_checkpoint.py \
  --checkpoint saved_models/pure_league_play_25M/checkpoint \
  --protocol benchmark \
  --preset pure_league_play
```

## Resume

```bash
RESUME_CURRICULUM_STAGE=league_training MLFLOW_RUN_ID=331f74404401435fbada77344ef3b8e5 \
  uv run train_battler.py --preset pure_league_play \
  --resume-checkpoint saved_models/pure_league_play_25M/checkpoint \
  --timesteps 30000000
```
