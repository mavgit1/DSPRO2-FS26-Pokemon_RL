# pure_league_pool_20team_13p7m

Final checkpoint from the pre-curriculum 20-team pool run (stopped before pool-staged restart).

| Field | Value |
|-------|--------|
| RLlib checkpoint | `checkpoint/` (copy of `step_13728072`) |
| Self-play weights | `selfplay_latest.pt` |
| Env steps at save | ~13,728,072 |
| Preset at train time | `pure_league_pool` (20 teams from start, old curriculum) |
| MLflow run | `53f146d60dcc40ea8e437814ca3fa473` |
| Training date | 2026-05-24 |
| Notes | Run ended on GPU/cublas crash; ~30% rolling WR in league phase |

## Load for eval

```bash
uv run python scripts/validate_checkpoint.py \
  --checkpoint saved_models/pure_league_pool_20team_13p7m/checkpoint \
  --protocol benchmark \
  --preset pure_league_play
```

## Resume (same architecture; not recommended for new pool curriculum)

```bash
uv run train_battler.py --preset pure_league_play \
  --resume-checkpoint saved_models/pure_league_pool_20team_13p7m/checkpoint \
  --timesteps 25000000
```
