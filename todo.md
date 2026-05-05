# TODO: Achieve >15% Win Rate vs Heuristic Opponents

## Current State (2026-05-03)

**Architecture:** 1-layer pre-norm transformer + learnable attention bias + LSTM
**Best result:** 10% win rate vs heuristic (sporadic), 65-85% vs random
**MLflow run:** `tasteful-fox-365` / `4fbec274deda4e2bb859ca52496291d8`

### What we've tried

| Experiment | CLS→opp_active | vs_heuristic | Verdict |
|------------|---------------|-------------|---------|
| 4-layer post-norm (baseline) | 12.5% | 7.5% | Attention collapse in layers 2-3 |
| 2-layer pre-norm | 0.0% | 0% | Worse — pre-norm didn't help |
| 2L pre-norm + attn bias | 12.5% | 0% | Fixed layer 0, layer 1 still collapsed |
| 1L + attn bias | 37.5% | 0-5% | Best attention, still no wins |
| 1L + attn bias + matchup reward | 62.5% | 0% | Strongest attention ever, still no wins |
| **1L + attn bias + matchup + action quality** | **?** | **0-10%** | Sporadic 10% peaks, no consistent improvement |

### Core finding

**Reward shaping (matchup + action quality) does not move the heuristic win rate.** The model
attends to the opponent but cannot translate that into winning play. The bottleneck is
fundamental — the policy cannot learn the multi-turn strategic reasoning needed to beat a
heuristic with reward shaping alone. 5.2M steps of training confirmed no improving trend.

---

## Completed

### Step 1: Cross-team attention bias (DONE)
- Learnable `attn_bias [num_layers, 13, 13]` initialized with +2.0 for cross-team pairs
- Result: CLS→opp_active went from 0% → 37.5% → 62.5%

### Step 3: Type matchup reward shaping (DONE)
- `_compute_matchup_quality()` scores best move type effectiveness vs opponent active
- Weight increased from 0.5 → 5.0 in curriculum stage
- Result: attention improved but heuristic win rate unchanged

### Step A: Increase matchup reward weight (DONE)
- Set `matchup_reward_weight=5.0` in curriculum stage
- No meaningful impact on heuristic win rate

### Step B: Action-level move effectiveness reward (DONE)
- `_compute_action_quality()` added to `PokemonBattleEnv` with offensive + defensive components
- Offensive: penalizes picking sub-optimal damaging moves (chosen vs best effectiveness)
- Defensive: rewards when our active resists (+0.5) or is immune (+1.0) to opponent's best known move
- Wired into `_compute_configured_delta_reward()` via `action_quality_weight=2.0`
- Config: `RewardConfig.action_quality_weight`, propagated through `CurriculumStageConfig.to_dict()`

### Step C: Defensive matchup reward (DONE)
- Integrated into `_compute_action_quality()` defensive component

### Step D: Train longer (DONE — no improvement)
- Ran standard preset for 5.2M steps (stopped early — no trend)
- Validation every 100k steps with smoke/fixed_paired/mirror protocols
- vs random: stable 65-90% throughout
- vs heuristic: 0-10% with no upward trend, sporadic 10% peaks at 314k and 3.66M
- **Verdict: more training does not help. The reward signal is insufficient.**

---

## Experiment: Action Quality Reward Training Results

**Config:** standard preset, 100% heuristic opponents, `matchup_reward_weight=5.0`, `action_quality_weight=2.0`
**Steps trained:** ~5.2M (of planned 10M, stopped early)

### Heuristic win rate over time (fixed_paired / mirror, 20 games each)

| Step | vs Random | vs Heuristic (FP) | vs Heuristic (Mirror) |
|------|-----------|-------------------|----------------------|
| 105k | 70-90% | 5% | 0% |
| 209k | 75-80% | 0% | 0% |
| 314k | 70-85% | **10%** | 0% |
| 419k | 85-90% | 0% | 0% |
| 524k | 80-85% | 5% | 0% |
| 628k | 60-70% | 0% | 5% |
| 733k | 70-85% | 5% | 0% |
| 838k | 60-70% | 5% | 0% |
| 943k | 85-95% | 0% | 5% |
| 1.05M | 45-70% | 0% | 0% |
| 1.15M | 65-75% | **10%** | 0% |
| 1.26M | 70-80% | 0% | 0% |
| 1.36M | 75% | 0% | 0% |
| 1.47M | 75-85% | 5% | 0% |
| 1.57M | 85-90% | 0% | 0% |
| 1.68M | 80-85% | 0% | 5% |
| 1.78M | 80% | 5% | 0% |
| 1.89M | 70% | 0% | 5% |
| 1.99M | 55-80% | 0% | 0% |
| 2.3M | 65% | 0% | 5% |
| 2.4M | 85% | 0% | 5% |
| 2.5M | 80% | 0% | 0% |
| 2.6M | 75% | 0% | 5% |
| 2.7M | 80% | 0% | 0% |
| 2.8M | 65% | 5% | 0% |
| 2.9M | 75% | 5% | 0% |
| 3.0M | 65% | 0% | 5% |
| 3.1M | 70% | 0% | **10%** |
| 3.2M | 70% | 0% | 5% |
| 3.5M | 55% | 5% | 5% |
| 3.7M | 75-80% | **10%** | **10%** |
| 3.9M | 90-95% | 5% | 0% |
| 4.1M | 60% | 0% | 0% |
| 4.3M | 70% | 5% | 0% |
| 4.4M | 70% | 0% | 0% |
| 4.8M | 65% | 0% | **10%** |
| 5.0M | 70% | 5% | 0% |
| 5.2M | 60% | 0% | 0% |

### Conclusion

**Reward shaping is not the bottleneck.** The action quality reward gives the model direct
signal about which move to pick, but heuristic win rate stays at 0-10% with no trend over
5.2M steps. The model can attend to the opponent and knows which moves are effective, but
cannot chain that into multi-turn winning strategies.

---

## Next Steps (Priority Order)

### Step G: Imitation learning from heuristic (HIGH PRIORITY)

**Why:** The heuristic player knows how to win. Instead of discovering winning strategies
through reward, directly learn from expert demonstrations.

**What to do:**
- Collect battle logs from heuristic vs heuristic games
- Add behavioral cloning loss on the CLS token output
- Or use DAgger-style online imitation

### Action quality refinements (LOW PRIORITY — reward shaping alone is insufficient)

#### Priority-aware SE exemption
If the chosen move has higher priority than the best-SE move, don't penalize.

#### STAB bonus awareness
Compare real damage (`base_power * effectiveness * 1.5(if STAB)`) not just type multiplier.

#### Switch-to-resist bonus
When switching, check if incoming mon has better defensive matchup than outgoing.

---

## Priority Order

1. **Step G** (imitation learning) — bypasses reward shaping entirely
2. Action quality refinements — low priority, reward shaping alone isn't enough


# TO FIX!:
Logging is fucked again win rate against heuristic and self is not being logged properly.

Playing against self has way too high win rate. Diagnose and fix opposing self.

Current stage 3 is way easier than 2. But is probably based on the weak self play.

Increase validation battle number too noisy of a signal

Current config seems good for some reason.