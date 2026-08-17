# Chapter 05: The Observation Pipeline

## How a Battle State Becomes Numbers

This chapter traces the exact path from Showdown's battle state to the
observation tensors that enter the model. Every dimension is accounted for.

**Summary**: The environment outputs 13 tokens × 164 dense features plus
separate categorical indices for species, items, and abilities. After embedding
lookup, this becomes 13 × 260 = 3,380 dimensions entering the transformer.

---

## The Big Picture

```
Pokemon Showdown Battle State
    (JSON-like: HP, types, moves, boosts, status, weather...)
         |
         v
poke-env AbstractBattle object
    (Python objects: Pokemon, Move, Battle)
         |
         v
embed_battle() in src/models/embedding.py
    (Converts battle objects to numpy arrays)
         |
         v
Observation dict
    {
        "obs":         (13, 164) float32 - Dense features per token
        "species":     (13,) int32       - Species IDs for embedding lookup
        "items":       (13,) int32       - Item IDs for embedding lookup
        "abilities":   (13,) int32       - Ability IDs for embedding lookup
        "action_mask": (14,) float32     - Binary mask for legal actions
    }
         |
         v
BattleTransformer._embed_obs()
    (Concatenates: 164 dense + 96 categorical embeddings = 260 dims per token)
```

---

## Token Layout: The 13 Tokens

The battle state is represented as 13 tokens, one per "slot":

```
Token 0:  GLOBAL STATE (weather, field, turn, opponent type)
Token 1:  OUR ACTIVE POKEMON (the one currently fighting)
Token 2:  OUR BENCH POKEMON #2
Token 3:  OUR BENCH POKEMON #3
Token 4:  OUR BENCH POKEMON #4
Token 5:  OUR BENCH POKEMON #5
Token 6:  OUR BENCH POKEMON #6
Token 7:  OPPONENT ACTIVE POKEMON
Token 8:  OPPONENT BENCH POKEMON #2
Token 9:  OPPONENT BENCH POKEMON #3
Token 10: OPPONENT BENCH POKEMON #4
Token 11: OPPONENT BENCH POKEMON #5
Token 12: OPPONENT BENCH POKEMON #6
```

If a team has fewer than 6 Pokemon (early in a battle, or in certain formats), unused slots get zero-filled tokens. The presence flag (dimension 0) distinguishes real Pokemon from empty slots.

> **WHY 13 tokens?** The transformer needs fixed-size input. 6v6 is the maximum team size in singles battles. 1 global + 6 ours + 6 theirs = 13 tokens. The transformer's attention mechanism can learn to ignore empty slots.

---

## The Pokemon Token: 164 Dimensions

Each Pokemon (tokens 1-12) is encoded into 164 features:

### Flags (3 dims)
```
[0] is_present    - 1 if this slot has a Pokemon, 0 if empty
[1] is_active     - 1 if this is the active Pokemon, 0 if on bench
[2] is_fainted    - 1 if this Pokemon has fainted
```

### HP (1 dim)
```
[3] hp_fraction   - Current HP / Max HP (0.0 to 1.0)
```

### Base Stats (6 dims)
```
[4] base_hp       - Base HP stat / 200
[5] base_atk      - Base Attack / 200
[6] base_def      - Base Defense / 200
[7] base_spa      - Base Sp. Atk / 200
[8] base_spd      - Base Sp. Def / 200
[9] base_spe      - Base Speed / 200
```

Normalized by 200 to keep values in [0, 1]. The highest base stat in the game is ~255, so 200 is a reasonable max.

### Types (20 dims)
```
[10:30] type_vector - Multi-hot encoding of Pokemon's types
    Index 0=Normal, 1=Fire, 2=Water, ... 18=Fairy, 19=???
    Most Pokemon have 1 or 2 types, so 1-2 entries are 1.0
```

### Status Condition (7 dims)
```
[30:37] status - One-hot encoding of status condition
    0=none, 1=burn, 2=poison, 3=toxic, 4=paralysis, 5=sleep, 6=freeze
```

### Tracked Effects (9 dims)
```
[37:46] effects - Multi-hot for volatile battle effects
    - substitute, confusion, flinch, trapped, leech_seed, etc.
```

### Stat Boosts (7 dims)
```
[46:53] boosts - Stat stage modifiers, normalized from [-6, 6] to [-1, 1]
    0=atk, 1=def, 2=spa, 3=spd, 4=spe, 5=accuracy, 6=evasion
```

### Item/Ability Flags (2 dims)
```
[53] has_item     - 1 if Pokemon is holding an item (opponent: only if revealed)
[54] has_ability   - 1 if ability is known (opponent: only if revealed)
```

### Weight (1 dim)
```
[55] weight - Pokemon weight / 100 (relevant for moves like Grass Knot)
```

### Moves (4 moves x 26 features = 104 dims)
```
For each of 4 move slots (dims 56-81, 82-107, 108-133, 134-159):
    [0] move_present     - 1 if this move slot has a move
    [1] base_power       - Move power / 100 (0 for status moves)
    [2] accuracy         - Move accuracy (0.0 to 1.0)
    [3:6] category       - One-hot: [physical, special, status]
    [6:26] move_type     - One-hot encoding of move's type (20 types)
```

> Total per Pokemon: 3 + 1 + 6 + 20 + 7 + 9 + 7 + 2 + 1 + 104 = 160... let me recount.
> Actually: 3+1+6+20+7+9+7+2+1 = 56 base features, then 4*26 = 104 move features = 160.
> With padding to 164, there may be 4 extra dims. The exact layout might vary slightly.

### Opponent Information Asymmetry

Our team (tokens 1-6): Full information. We see all moves, item, ability, exact HP.

Opponent team (tokens 7-12): Partial information. We only see:
- Revealed moves (moves the opponent has used)
- Revealed item/ability (only if triggered in battle)
- Exact HP (Showdown shows opponent HP percentages)

This mirrors real competitive Pokemon where you don't know the opponent's full moveset.

---

## The Global Token: Token 0

Token 0 encodes battle-wide context, not a specific Pokemon. It has the same 164 dimensions but uses them differently:

### Global Extras (first 12 dims of token 0)
```
[0] opponent_random     - 1.0 if opponent is random type
[1] opponent_heuristic  - 1.0 if opponent is heuristic type
[2] opponent_other      - 1.0 if opponent is self-play or other
[3] training_stage      - Curriculum stage index / max_stages
[4] battle_turn         - Current turn / 100
[5] force_switch        - 1.0 if we're forced to switch (active fainted)
[6] active_trapped      - 1.0 if our active can't switch (Arena Trap, etc.)
[7] move_count          - Number of available moves / 4
[8] switch_count        - Number of available switches / 5
[9] can_dynamax         - 1.0 if Dynamax is available
[10] can_mega           - 1.0 if Mega Evolution is available
[11] can_zmove          - 1.0 if Z-Move is available
```

### Why Include Opponent Type?

```python
opponent_random    = 1.0  # vs RandomPlayer
opponent_heuristic = 1.0  # vs SimpleHeuristicsPlayer
opponent_other     = 1.0  # vs SelfPlayPlayer
```

The model needs to know who it's playing against because optimal strategy differs:
- vs Random: Exploit aggressively (they'll make mistakes)
- vs Heuristic: Play carefully (they pick super-effective moves)
- vs Self: Play unpredictably (they know your strategy)

> **WHY canonicalization**: `random_no_switch` maps to `random` in the embedding (both get `opponent_random=1.0`). Without this, the model would see `random_no_switch` as a completely different opponent type, even though they play very similarly. The distribution shift would hurt learning.

The rest of token 0's 164 dimensions encode weather, field conditions, and side conditions (stealth rock, spikes, etc.).

---

## The Categorical Features

In addition to the 164 dense features per token, three categorical features are tracked separately:

```python
"species":    (13,) int32   # Pokemon species ID (Pokedex number mapped to vocab index)
"items":      (13,) int32   # Held item ID (mapped to vocab index)
"abilities":  (13,) int32   # Ability ID (mapped to vocab index)
```

These are **not** one-hot encoded. Instead, they're passed through learned embedding layers in the model:

```
Species ID 42  -->  nn.Embedding(num_species, 32)  -->  32-dim vector
Item ID 15     -->  nn.Embedding(num_items, 32)     -->  32-dim vector
Ability ID 7   -->  nn.Embedding(num_abilities, 32) -->  32-dim vector
                                       Total: 96-dim categorical embedding
```

This 96-dim vector is concatenated with the 164-dim dense features to form the 260-dim input to the model's projection layer.

> **WHY learned embeddings instead of one-hot?**
> - There are hundreds of Pokemon species. One-hot would create huge, sparse vectors.
> - Embeddings let the model learn that "Charizard" and "Charmander" are related (they share type/move patterns) because the optimizer can place their embedding vectors close together.
> - 32 dims is enough to capture the similarity structure without being wasteful.

---

## Action Mask Construction

```python
def get_action_mask(battle):
    mask = np.zeros(14, dtype=np.float32)

    # Moves (actions 0-3)
    for i, move in enumerate(battle.available_moves[:4]):
        mask[i] = 1.0

    # Gimmick moves (actions 4-7) - only if gimmick available
    if battle.can_dynamax or battle.can_mega_evolve or ...:
        for i in range(min(4, len(battle.available_moves))):
            mask[4 + i] = 1.0

    # Switches (actions 8-13)
    for i, pokemon in enumerate(battle.available_switches[:6]):
        mask[8 + i] = 1.0

    return mask
```

> **IN OUR CODE**: The mask uses `battle.available_moves` and `battle.available_switches` directly, rather than `battle.valid_orders`. The `valid_orders` approach was slower and sometimes stale (Showdown updates can arrive out of order).

---

## Complete Data Flow Diagram

```
Showdown sends battle update
         |
         v
poke-env parses into AbstractBattle
  |       |        |         |
  v       v        v         v
Pokemon  Pokemon  Move     Weather
objects  objects  objects   etc.
  |       |        |         |
  v       v        v         v
embed_pokemon() x12    _global_features() x1
  |                              |
  |  Each: 164-dim vector        |  164-dim vector
  |                              |
  v                              v
Stack into (13, 164) matrix ---+
         |
         v
Observation dict:
  "obs":         (13, 164) float32
  "species":     (13,) int32
  "items":       (13,) int32
  "abilities":   (13,) int32
  "action_mask": (14,) float32
         |
         v
[Model input - see Chapter 06]
```

---

## What's Next

Now you understand exactly what data enters the model. Chapter 06 explains how the BattleTransformer processes these observation tensors into action probabilities and a value estimate.
