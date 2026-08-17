# Chapter 09: Self-Play

## How the Agent Plays Against Itself

Self-play is one of the most powerful training techniques in RL.
AlphaGo, AlphaStar, and OpenAI Five all used it extensively.
Here's how our Pokemon agent does it.

---

## CONCEPT: Why Self-Play?

When you only train against fixed opponents (random, heuristic), the agent learns to exploit their specific weaknesses. But it doesn't learn general strategies.

Self-play creates a training loop:
1. Agent plays against a copy of itself
2. As the agent improves, its opponent improves too
3. The agent must constantly find new strategies to beat its improving self
4. This creates an ever-escalating arms race

```
Agent v1 beats Agent v1 (random initialization) -> learns basic strategies
Agent v2 beats Agent v1 -> learns counter-strategies
Agent v3 beats Agent v2 -> learns counter-counter-strategies
...and so on
```

---

## The SelfPlayPlayer

```python
class SelfPlayPlayer(Player):
    def __init__(self, weights_path, model_config):
        self.model = PokemonTransformerModel(...)
        self.weights_path = weights_path
        self._last_mtime = 0  # Track file modification time
        self._lstm_cache = {}  # Per-battle LSTM state
        self._diag = DiagnosticsAccumulator()
```

### Weight Loading System

The trainer writes `checkpoints/selfplay_latest.pt` *before* each PPO iteration **only when the active mix can sample `self` or `historical`**. `begin_episode()` loads that snapshot once. Inference is `eval()` + `no_grad()`, so the CPU copy is not mutated and must not be `load_state_dict`'d again every turn.

If `freeze_weights_per_episode` is False, `_try_load_weights()` may reload when the file mtime changes.

### The CRITICAL Absolute Path Bug

> **This was the single most costly bug in the project.**

```python
# WRONG (breaks silently):
selfplay_weights_path = "checkpoints/selfplay_latest.pt"

# CORRECT:
selfplay_weights_path = os.path.abspath("checkpoints/selfplay_latest.pt")
```

Ray workers run from `/tmp/ray/session_*/working_dir_files/_ray_pkg_*/`. When the `SelfPlayPlayer` tries to load `"checkpoints/selfplay_latest.pt"`, it looks for that relative path from Ray's working directory, which doesn't exist. The `try/except` silently caught the error, and the self-play opponent played with random initialization for months.

The fix: always use `os.path.abspath()` for ANY file path passed to Ray workers.

---

## Weight Export (Trainer Side)

Every training iteration **that uses self/historical opponents**, the trainer exports model weights:

```python
if self._selfplay_snapshot_needed():
    self._export_selfplay_weights()
```

> **WHY every such iteration?** If we only export at checkpoints (every 150k steps), the self-play opponent would be stale for thousands of battles. Skip the write when the mix is random/heuristic-only (league warmup). Export immediately when promoting into a mix that needs the snapshot.

---

## Action Selection in Self-Play

```python
def choose_move(self, battle):
    """Called by poke-env when it's the opponent's turn to act."""
    # 1. Frozen snapshot already loaded in begin_episode(); no per-turn restore

    # 2. Get observation
    obs = embed_battle(battle, opponent_type="self")
    mask = get_action_mask(battle)

    # 3. Forward pass through model
    with torch.no_grad():
        logits, _ = self.model(obs, mask)

    # 4. Temperature sampling (NOT argmax)
    probs = F.softmax(logits / temperature, dim=-1)  # temperature = 0.8
    probs[~mask.bool()] = 0
    action = torch.multinomial(probs, 1)

    # 5. Convert to Showdown order
    return action_to_order(action, battle)
```

### Why Temperature Sampling (Not Argmax)?

```python
temperature = 0.8  # Slightly sharper than uniform, but not deterministic
```

- **Argmax**: Always picks the best action. Deterministic. The training agent can learn to exploit the exact pattern.
- **Uniform random**: Too random. Doesn't provide a meaningful challenge.
- **Temperature 0.8**: Sharpens the distribution slightly (favors better actions) but maintains randomness. The training agent can't predict the exact move.

```
With temperature=0.8:
  Good move (raw prob 0.3): boosted to ~0.35
  Bad move (raw prob 0.05): reduced to ~0.03
  Total: still somewhat random, but biased toward good play
```

---

## The NaN Bug (PyTorch 2.10)

### What Happened

When self-play was first enabled, the opponent lost 90%+ of battles even after the weight loading was fixed. Investigation revealed:

1. The model uses a learnable attention bias: `self.attn_bias = nn.Parameter(...)`
2. This bias is a float tensor passed to `TransformerEncoderLayer.forward(src_mask=bias)`
3. PyTorch 2.10 has a bug: float `src_mask` triggers NaN in the fused attention kernel
4. NaN propagated through the forward pass
5. NaN logits reached `torch.multinomial()`, which crashed
6. The fallback caught the crash and returned random actions

### The Fix

Replace the standard `layer(x, src_mask=bias)` call with manual norm-first decomposition:

```python
# BEFORE (causes NaN):
output = transformer_layer(x, src_mask=bias)

# AFTER (works correctly):
def _transformer_forward(self, x, attn_bias):
    for layer in self.layers:
        # Step 1: LayerNorm + Self-Attention
        normed = layer.norm1(x)
        attn_out, _ = layer.self_attn(
            normed, normed, normed,
            attn_mask=attn_bias  # Works correctly with attn_mask param
        )
        x = x + attn_out  # Residual

        # Step 2: LayerNorm + FFN
        normed = layer.norm2(x)
        ffn_out = layer.linear2(
            layer.activation(layer.linear1(normed))
        )
        x = x + ffn_out  # Residual
    return x
```

> **DO NOT REVERT THIS.** The NaN bug is specific to PyTorch 2.10's fused kernel. Until it's fixed upstream, always use manual decomposition.

---

## Self-Play Diagnostics

To catch self-play issues early, a comprehensive diagnostics system was built:

### What's Tracked

```python
class DiagnosticsAccumulator:
    weight_load_count: int          # How many times weights were reloaded
    fallback_count: int             # How many times it fell back to random
    action_mapping_fallback: int    # Action mapping failures
    action_histogram: dict          # Distribution of chosen actions
    top_prob_sum: float             # Sum of highest action probabilities
    entropy_sum: float              # Sum of policy entropy
    valid_action_count: float       # Average number of valid actions
```

### Diagnostic Flow

```
SelfPlayPlayer._diag (accumulates per-turn)
         |
         v
pop_diagnostics() (returns and resets accumulator)
         |
         v
CurriculumSingleAgentWrapper.pop_selfplay_diagnostics()
         |
         v
env_bridge.collect_selfplay_diagnostics(algo)  # Aggregates across all workers
         |
         v
trainer._collect_and_log_selfplay_diagnostics()
         |
         +---> logs/selfplay_diagnostics.log (JSON lines)
         +---> MLflow metrics
```

### What Healthy Self-Play Looks Like

```
fallback_rate: 0%           # No random fallbacks
mapping_fallback_rate: 0%   # All actions convert cleanly
avg_top_prob: 0.21 -> 0.47  # Agent becoming more confident over training
avg_entropy: ~1.5-2.0       # Still exploring, not collapsed
valid_action_count: ~5-6    # Typical number of legal actions
```

### Diagnostics Script

```bash
uv run scripts/diagnose_selfplay.py --preset standard --timesteps 500000
```

This runs a 500k-step training session with 30% self-play that never promotes from stage 0. It's a diagnostic tool specifically for exercising and inspecting the self-play system.

---

## Opponent Type in Observations

When playing against self, the observation includes `opponent_other=1.0`:

```python
# In embedding.py
def _canonical_opponent_type(opponent_type):
    if opponent_type == "random" or opponent_type == "random_no_switch":
        return "random"
    elif opponent_type == "heuristic":
        return "heuristic"
    else:
        return "other"  # self-play maps here
```

The model sees three opponent type flags:
```
[0] opponent_random     = 0.0  (not random)
[1] opponent_heuristic  = 0.0  (not heuristic)
[2] opponent_other      = 1.0  (self-play!)
```

> **WHY**: The model needs to know it's playing against itself because self-play requires different strategy than fixed opponents. Against random, exploit aggressively. Against self, play unpredictably.

---

## What's Next

Chapter 10 covers the validation system - how we measure whether the agent is actually getting better.
