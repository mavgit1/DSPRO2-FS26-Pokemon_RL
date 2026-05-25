# Team pool curriculum manifests

Subsets of `data/validation/gen8_random_battle_team_pairs.json` for staged training.

| File | Teams | Use |
|------|-------|-----|
| `gen8_pool_3.json` | 3 | First pool stage |
| `gen8_pool_5.json` | 5 | Second pool stage |
| `gen8_pool_10.json` | 10 | Third pool stage |
| `gen8_pool_20.json` | 20 | Full validation set (same teams as full manifest) |

Regenerate from the validation manifest:

```bash
python scripts/build_team_pool_manifest.py
```

Preset `pure_league_play` sets `CurriculumStageConfig.team_pool_manifest` per stage: warmup and `heuristic_bridge` on 3 teams, then pool stages at **65%** rolling win rate with the heuristic-tactics mix (65% heuristic, 25% self, 5% random, 5% random-no-switch), then `league_training` on 20 teams.

Override the default manifest path in env config:

```python
env=replace(..., team_pool_manifest="data/teams/pool_curriculum/gen8_pool_5.json")
```

Or per curriculum stage via `team_pool_manifest` on `CurriculumStageConfig`.
