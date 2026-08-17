# Chapter 13: Key Decisions

## Every Major Design Choice and Why

This chapter collects all the important decisions in one place.
Each entry explains what was chosen, what alternatives were considered,
and what went wrong before arriving at the current solution.

---

## 1. Transformer Architecture

**Decision**: 2-layer transformer with 512 hidden dim, 8 heads, LSTM temporal memory.

**Alternatives considered**:
- Deeper transformers (4-6 layers): Vanishing signals, slower training, no improvement
- No LSTM: Agent couldn't remember previous turns (opponent used Protect? switching patterns?)
- MLP (no attention): Couldn't model relationships between team Pokemon

**Key insight**: Depth >2 caused vanishing signals. The attention bias (+2.0 CLS-to-opponent) was more impactful than adding layers.

---

## 2. Action Space: 14 Discrete Actions

**Decision**: Compressed to 14 actions (4 moves + 4 gimmick + 6 switches).

**Alternatives considered**:
- Full Showdown action space (~500+ possible orders): Too sparse, most never used
- Separate move/switch heads: Added complexity, no benefit

**Key insight**: Action masking makes the 14-action space effectively ~5-6 legal actions per turn. This is small enough for PPO to explore efficiently.

---

## 3. Observation: 13 Tokens × 260 Dims (164 Dense + 96 Categorical)

**Decision**: Fixed-size token representation with learned embeddings for categorical features.

**Structure**:
- 13 tokens × 164 dense features = 2,132 floats (HP, types, moves, boosts, etc.)
- 13 categorical indices each for: species (1,516 vocab), items (584 vocab), abilities (315 vocab)
- After embedding lookup: 13 × (164 + 32 + 32 + 32) = 13 × 260 = 3,380 dims to transformer
- Plus action mask: 14 dims

**Alternatives considered**:
- Flat vector (no token structure): Couldn't model Pokemon relationships
- Variable-length sequences: Incompatible with batch processing
- Graph neural network: More expressive but harder to train
- One-hot categorical features: Too sparse (1,500+ species)

**Key insight**: The transformer token pattern naturally fits the problem - 12 Pokemon slots + 1 global = 13 tokens. Separate categorical embeddings let the model learn that similar Pokemon (evolution family, same type) have similar representations. This is impossible with one-hot encoding.

---

## 4. Reward Scale: 0.05

**Decision**: Multiply all rewards by 0.05 before returning.

**Why**: Without this, returns in [-15, +15] were impossible for the value function to predict. Explained variance went from 0.1 to 0.7+ after scaling to [-0.75, +0.75].

**Lesson**: RLlib normalizes advantages but NOT returns. Always check what your value function is actually trying to predict.

---

## 5. Gradient Clipping: 5.0

**Decision**: `grad_clip = 5.0`

**History**: Was 0.5 (standard default). Total gradient norm was ~65. All gradients scaled to 0.77% of their original value. The transformer and embedding layers received effectively zero gradient. Months of training with no learning in those layers.

**Lesson**: Always log gradient norms. If `total_norm >> grad_clip`, you're starving your model.

---

## 6. Gamma: 0.97

**Decision**: Discount factor of 0.97 (~33-step effective horizon).

**History**: 0.99 (standard) gave too long a horizon (~100 steps for 25-turn battles). 0.95 was too short. 0.97 hits the sweet spot where the agent cares about ~33 steps ahead, which covers the important strategic decisions in a battle.

---

## 7. Entropy Coefficient: 0.013

**Decision**: Very low entropy bonus, found by Optuna sweep.

**History**:
- 0.005: Negligible. Policy collapsed.
- 0.05: Too much exploration. Agent couldn't commit to strategies.
- 0.2: Pure chaos. No convergence.
- 0.013: Optuna found this. Enough exploration to discover, enough commitment to exploit.

---

## 8. PPO Clip Param: 0.08

**Decision**: Much tighter than the standard 0.2.

**Why**: Pokemon has a complex state space with sparse rewards. Large updates (clip=0.2) destabilized learning - the policy would swing wildly between strategies. Tight clipping (0.08) forces gradual, stable improvement.

---

## 9. Attention Bias (Learnable)

**Decision**: Add hand-crafted priors to attention scores (+2.0 CLS-to-opponent, etc.)

**Why**: Standard self-attention treats all token pairs equally. In Pokemon, the most important relationship is "our active vs their active." The bias seeds the model with this knowledge so it doesn't have to discover it from scratch.

**Gotcha**: PyTorch 2.10 NaN bug with float `src_mask`. Must use manual norm-first decomposition.

---

## 10. Curriculum Learning

**Decision**: 2-stage curriculum (warmup -> mixed_final).

**Alternatives considered**:
- Single opponent: Agent overfits to one opponent's weaknesses
- 5-stage curriculum: Too complex, hard to tune thresholds
- Self-play only: Agent needs some baseline opponents to learn basics

**Key insight**: The warmup stage with 30% self-play is crucial. It exposes the agent to self-play early (when it's still weak and easy to beat) rather than suddenly introducing it later.

---

## 11. Self-Play with Temperature Sampling

**Decision**: Self-play opponent uses temperature=0.8 sampling.

**Alternatives considered**:
- Argmax: Deterministic, easily exploitable
- Uniform random: No challenge
- Temperature 0.5: Too sharp, almost deterministic
- Temperature 1.0: Same as training distribution

**Why 0.8**: Slightly sharper than training, favoring better actions while maintaining unpredictability.

---

## 12. In-Process Validation

**Decision**: Run validation using the live `algo` object, not a subprocess.

**Why**: Subprocess validation had 30-60s overhead (Ray init, checkpoint restore) per evaluation. With validation every 100k steps, this added significant overhead. In-process validation uses the existing model in memory.

---

## 13. 8 Showdown Servers

**Decision**: Run 8 Pokemon Showdown instances on ports 8000-8007.

**Why**: Showdown is single-threaded Node.js. One instance can handle ~24 concurrent battles but starts bottlenecking beyond that. With 192 parallel environments, 8 servers distribute the load evenly.

---

## 14. Species/Item/Ability Embeddings (32 dims each)

**Decision**: Learn 32-dim embeddings for each categorical feature.

**Alternatives considered**:
- One-hot encoding: Too sparse (1000+ species)
- Larger embeddings (64/128 dims): More parameters, slower, no improvement
- No embeddings (only dense features): Model couldn't distinguish between species

**Key insight**: Embeddings let the model learn that similar Pokemon (evolution family, same type) have similar representations. This is impossible with one-hot.

---

## 15. Separate Policy and Value Heads

**Decision**: Shared transformer backbone, separate 256-dim heads.

**Why**: The policy needs to output 14 action probabilities while the value needs to output 1 scalar. They share the backbone (most of the learning happens there) but have separate final layers for their different objectives.

---

## 16. No Dynamax / No Terastallize

**Decision**: Custom Showdown formats that disable gimmicks.

**Why**: These mechanics add massive complexity (Dynamax doubles HP, adds secondary effects; Tera changes types). They're mechanical advantages rather than strategic ones. Removing them focuses the agent on core battle strategy.

---

## 17. Wilson Confidence Intervals for Validation

**Decision**: Report 95% Wilson CIs alongside win rates.

**Why**: With only 50 episodes per opponent, win rates are noisy. Wilson CIs tell us whether a 65% win rate is significantly different from 50% or just noise. Essential for making reliable training decisions.

---

## 18. Reward Composition

**Decision**: Terminal (+/-10) dominates, HP delta (2.0) secondary, shaping (0.2, 0.3) minimal.

**Why**: The reward hierarchy ensures the agent optimizes for the true objective (winning). Shaping rewards are intentionally weak - they provide directional hints but can't create local optima.

---

## 19. Delta Rewards (Not Absolute)

**Decision**: Reward based on change from previous turn, not current state.

**Why**: "My HP went from 80% to 60%" is actionable. "I have 60% HP" is ambiguous (could be winning or losing). Delta rewards create a natural comparison signal.

---

## 20. Opponent Canonicalization in Embeddings

**Decision**: `random_no_switch` maps to `random` in observation features.

**Why**: Without this, the model treats `random_no_switch` as a completely different opponent type, even though it plays almost identically. The feature distribution shift would hurt learning.

---

## Summary: What Went Wrong Most Often

1. **Reward scaling** (months of low explained variance)
2. **Gradient clipping** (months of no transformer learning)
3. **Self-play weight loading** (absolute path bug, months of random self-play)
4. **PyTorch NaN bug** (self-play always losing due to NaN logits)
5. **Per-step reward removal** (gradient cancellation, no learning)
6. **Reward hacking** (too-strong shaping rewards, optimizing proxy instead of winning)

**Takeaway**: In RL, the reward function and hyperparameters matter more than the model architecture. Most debugging time was spent on rewards and PPO params, not on the transformer.
