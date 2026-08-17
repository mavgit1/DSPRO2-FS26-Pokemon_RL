# Chapter 06: The Model Architecture

## How Numbers Become Decisions

This chapter covers the BattleTransformer - the neural network that takes
the observation tensors (13 tokens × 260 dims after embedding) and outputs action probabilities and a value estimate.

---

## The Big Picture

```
Input: Observation dict
  "obs":         (batch, 13, 164) float32
  "species":     (batch, 13) int32
  "items":       (batch, 13) int32
  "abilities":   (batch, 13) int32
  "action_mask": (batch, 14) float32
         |
         v
[1] Embedding Layer
    Dense features (164) + Categorical embeddings (96) = 260 dims
    Project to hidden_dim (512)
         |
         v
[2] Position & Role Embeddings
    Add structural information (which token is active, which is opponent)
         |
         v
[3] Transformer Encoder (2 layers)
    Self-attention between all 13 tokens
    Learns relationships (e.g., "our Fire type vs their Grass type")
         |
         v
[4] Extract CLS Token (token 0)
    This single 512-dim vector summarizes the entire battle state
         |
         +---> [5a] Policy Head: 512 -> 256 -> 14 logits
         |        Apply action mask -> softmax -> action probabilities
         |
         +---> [5b] Value Head: 512 -> 256 -> 1 scalar
                  "How good is this position?" (expected total reward)
```

---

## Layer 1: Embedding

### Categorical Embeddings

```python
self.species_embedding = nn.Embedding(num_species, 32, padding_idx=0)
self.item_embedding    = nn.Embedding(num_items, 32, padding_idx=0)
self.ability_embedding = nn.Embedding(num_abilities, 32, padding_idx=0)
```

Each ID (species, item, ability) is looked up in a learned embedding table:
- Species: ~1000 entries x 32 dims
- Items: ~500 entries x 32 dims
- Abilities: ~300 entries x 32 dims

`padding_idx=0` means ID 0 (empty slot) maps to a zero vector.

### Input Projection

```python
self.input_proj = nn.Linear(164 + 96, hidden_dim)
# 164 (dense) + 96 (32*3 categorical) = 260 -> 512
```

The dense features (164) and categorical embeddings (96) are concatenated, then projected to the model's hidden dimension (512).

> **WHY**: The model needs to learn relationships between dense features (HP, stats) and categorical features (species, items). The projection layer learns to fuse them into a unified representation.

---

## Layer 2: Position & Role Embeddings

After projection, each token gets additional structural information:

### Role Embeddings

```python
self.role_embedding = nn.Embedding(5, hidden_dim)
# Roles: CLS, our_active, our_bench, opp_active, opp_bench
```

| Token | Role | Embedding |
|-------|------|-----------|
| 0 | CLS (global) | role_emb[0] |
| 1 | Our active | role_emb[1] |
| 2-6 | Our bench | role_emb[2] |
| 7 | Opponent active | role_emb[3] |
| 8-12 | Opponent bench | role_emb[4] |

These are **added** to the token embeddings (not concatenated), following the standard transformer pattern.

> **WHY role embeddings?** The transformer's attention mechanism treats all tokens equally by default. Role embeddings tell it "this is our active Pokemon, that's the opponent's bench." Without them, the model would need to infer positions from the content alone.

---

## Layer 3: Transformer Encoder

### CONCEPT: What is Self-Attention?

Self-attention lets each token "look at" every other token and decide how much to care about it. For Pokemon battles, this means:

- Our active Pokemon can attend strongly to the opponent's active Pokemon ("What am I fighting?")
- Our bench Pokemon can attend to the opponent's type coverage ("Should I switch?")
- The CLS token attends to everything to build a global summary

Mathematically:
```
Attention(Q, K, V) = softmax(Q * K^T / sqrt(d)) * V
```
Where Q (query), K (key), V (value) are linear projections of the input.

### Our Transformer Configuration

```python
encoder_layer = nn.TransformerEncoderLayer(
    d_model=512,          # Hidden dimension
    nhead=8,              # 8 attention heads (64 dims per head)
    dim_feedforward=2048, # FFN: 512 -> 2048 -> 512
    dropout=0.0068,       # Very low dropout
    activation='gelu',    # GELU activation (smoother than ReLU)
    batch_first=True,     # Input shape: (batch, seq, features)
)
```

### Attention Bias

The most important non-standard feature. We add learnable biases to the attention scores:

```python
self.attn_bias = nn.Parameter(torch.zeros(num_layers, num_heads, 13, 13))
# Shape: (2 layers, 8 heads, 13 tokens, 13 tokens)
```

Initialized with hand-crafted priors:
```
CLS -> opp_active:   +2.0  (global token should strongly attend to what we're fighting)
our_active -> opp_active: +2.0  (our fighter should watch the opponent)
opp_active -> our_active: +1.0  (opponent's perspective on us)
opp_active -> our_bench:  +0.5  (opponent sizing up our switches)
CLS -> opp_bench:     +0.5  (global awareness of opponent's options)
```

> **WHY**: Standard self-attention treats all pairs equally. In Pokemon, some relationships are much more important than others. The attention bias seeds the model with this knowledge, so it doesn't have to learn it from scratch.

### The NaN Bug (Critical)

> **DO NOT** use `layer(x, src_mask=float_bias)` with PyTorch 2.10. It produces NaN due to a bug in the fused attention kernel.

Instead, we use manual norm-first decomposition:

```python
def _transformer_forward(self, x, attn_bias):
    for layer in self.transformer_layers:
        # Manual norm-first instead of layer(x, src_mask=bias)
        # 1. Layer norm
        normed = layer.norm1(x)
        # 2. Self-attention with bias
        attn_out = layer.self_attn(normed, normed, normed, attn_mask=bias)
        # 3. Residual connection
        x = x + attn_out[0]
        # 4. FFN with norm
        normed = layer.norm2(x)
        x = x + layer.linear2(layer.activation(layer.linear1(normed)))
    return x
```

This was the root cause of self-play losing 90%+ of battles: NaN logits -> random fallback.

---

## Layer 4: CLS Token Extraction

```python
cls_output = x[:, 0, :]  # Shape: (batch, 512)
```

Token 0 (the global/CLS token) is extracted. Through the transformer's attention, it has aggregated information from all other tokens - it "sees" the entire battle state.

> **WHY CLS token?** It's a standard transformer pattern. Instead of pooling all tokens (which loses positional information), we designate one token as the "summary" token. The attention mechanism naturally learns to use it for aggregation.

---

## Layer 5: Policy and Value Heads

### Policy Head

```python
self.policy_head = nn.Sequential(
    nn.Linear(hidden_dim, 256),
    nn.GELU(),
    nn.Linear(256, num_actions),  # 14 actions
)
```

Output: 14 raw logits. Then action masking is applied:

```python
logits[~action_mask.bool()] = -1e8  # Mask invalid actions
probs = F.softmax(logits, dim=-1)    # Convert to probabilities
```

The agent samples from this probability distribution during training (exploration) and takes the argmax during evaluation (exploitation).

### Value Head

```python
self.value_head = nn.Sequential(
    nn.Linear(hidden_dim, 256),
    nn.GELU(),
    nn.Linear(256, 1),
)
```

Output: A single scalar estimating the expected total (discounted) reward from this position. This is used by PPO to compute advantages (how much better an action was than expected).

> **WHY separate heads?** Policy and value serve different purposes:
> - Policy: "What should I do?" (categorical distribution)
> - Value: "How good is my position?" (scalar estimate)
>
> They share the transformer backbone but have separate final layers because their output formats and learning dynamics are different.

---

## The LSTM Layer (Temporal Memory)

```python
self.lstm = nn.LSTM(
    input_size=hidden_dim,
    hidden_size=lstm_hidden,  # 512
    batch_first=True,
)
```

The LSTM sits between the CLS extraction and the heads:

```
CLS token (512)
    |
    v
LSTM (hidden_size=512)
    | - Maintains hidden state across turns
    | - Remembers what happened in previous turns
    v
Policy/Value heads
```

### State Management

```python
def get_initial_state(self):
    """Return zeros for the initial LSTM state."""
    return [torch.zeros(1, lstm_hidden), torch.zeros(1, lstm_hidden)]
    # [hidden_state, cell_state]
```

The LSTM state persists across turns within a battle but resets between battles.

> **WHY LSTM?** A single-turn observation doesn't tell the full story. The opponent used Protect last turn? They probably won't use it again. Their Pokemon took damage but is still alive? How much? The LSTM gives the model a memory of previous turns.

### Per-Battle State Cache (Self-Play)

For the self-play opponent, LSTM state is cached per battle:

```python
class SelfPlayPlayer:
    def __init__(self):
        self._lstm_cache = {}  # battle_id -> (h, c)

    def _get_lstm_state(self, battle):
        return self._lstm_cache.get(battle.battle_tag, (zeros, zeros))
```

> **WHY**: Without this, the self-play opponent started each turn with zero LSTM state, forgetting everything from previous turns in the same battle. This was another bug that caused self-play to play terribly.

---

## Complete Forward Pass Diagram

```
Input dict:
  obs:         (B, 13, 164)
  species:     (B, 13)
  items:       (B, 13)
  abilities:   (B, 13)
  action_mask: (B, 14)
       |
       v
[Categorical Embeddings]
  species_emb:   (B, 13, 32)
  items_emb:     (B, 13, 32)
  abilities_emb: (B, 13, 32)
       |
       v
[Concatenate] -> (B, 13, 164+96) = (B, 13, 260)
       |
       v
[Input Projection] -> (B, 13, 512)
       |
       v
[+ Role Embeddings] -> (B, 13, 512)
       |
       v
[Transformer x2] with attention bias -> (B, 13, 512)
       |
       v
[Extract Token 0] -> (B, 512)
       |
       v
[LSTM] with state -> (B, 512)
       |
       +---> [Policy Head] -> (B, 14) logits
       |        [Apply mask] -> masked logits
       |        [Softmax] -> action probabilities
       |
       +---> [Value Head] -> (B, 1) value estimate
```

Where B = batch size.

---

## Model Size

```
Component             Parameters
-------------------   ----------
Input Projection      260 * 512 + 512 = ~133k
Species Embedding     ~1000 * 32 = ~32k
Item Embedding        ~500 * 32 = ~16k
Ability Embedding     ~300 * 32 = ~10k
Role Embedding        5 * 512 = ~2.5k
Attention Bias        2 * 8 * 13 * 13 = ~2.7k
Transformer x2        2 * (4 * 512^2 + 2 * 512 * 2048) = ~4.7M
LSTM                  4 * 512 * 512 = ~1.0M
Policy Head           512*256 + 256*14 = ~134k
Value Head            512*256 + 256*1 = ~131k
-------------------   ----------
TOTAL                 ~6.2M parameters
```

This is a small model by modern standards (GPT-4 has ~1.7T). The constraint is that it needs to run inference fast enough for real-time battle decisions.

---

## What's Next

Now you understand the model. Chapter 07 explains PPO - the algorithm that trains this model by collecting battle experience and updating the weights.
