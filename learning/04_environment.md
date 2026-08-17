# Chapter 04: The Environment

## How a Pokemon Battle Becomes an RL Problem

The environment is the bridge between Pokemon Showdown (the battle simulator)
and the RL algorithm. It translates battles into observations, actions, and rewards.

---

## The Layer Stack

```
Pokemon Showdown (Node.js server)       <-- Actual battle simulation
         |
         v
poke-env (Python library)               <-- HTTP/WebSocket interface to Showdown
         |
         v
PokemonBattleEnv (our code)             <-- Gymnasium-compatible wrapper
         |
         v
CurriculumSingleAgentWrapper (our code) <-- Opponent sampling, curriculum
         |
         v
Ray Worker                              <-- Parallel execution
         |
         v
PPO Algorithm                           <-- Collects observations, returns actions
```

---

## CONCEPT: What is a Gymnasium Environment?

A Gymnasium environment implements a standard interface for RL:

```python
obs, info = env.reset()       # Start a new episode (battle)
obs, reward, done, truncated, info = env.step(action)  # Take one step (one turn)
```

The environment must define:
- **Observation space**: What the agent sees (shape and type)
- **Action space**: What the agent can do
- **Step function**: Given an action, return (new_obs, reward, done, truncated, info)
- **Reset function**: Start a new episode

Our environment maps these to Pokemon battles:
- **Reset** = Start a new battle on Showdown
- **Step** = Choose a move/switch, wait for opponent, return new state
- **Observation** = The battle state (HP, types, moves, etc.) as tensors
- **Reward** = How well the turn went (HP change, fainting, type effectiveness)
- **Done** = Battle ended (win, loss, or max turns reached)

---

## The poke-env Library

`poke-env` is a Python library that:
1. Connects to a Pokemon Showdown server via WebSocket
2. Receives battle events (damage, fainting, switches)
3. Provides a Python API to query battle state
4. Allows you to submit moves/switches

It handles all the networking and Showdown protocol parsing. Without it, you'd need to implement the entire Showdown WebSocket protocol yourself.

### How poke-env Works

```
Our Code                    poke-env                    Showdown Server
   |                           |                              |
   | player.choose_move()      |                              |
   |-------------------------->|  format BattleOrder          |
   |                           |----------------------------->|
   |                           |  wait for opponent           |
   |                           |<-----------------------------|
   |                           |  parse events                |
   | return BattleOrder        |                              |
   |<--------------------------|                              |
   |                           |                              |
   | (battle state updated)    |                              |
   | battle.active_pokemon     |                              |
   | battle.opponent_active    |                              |
   | battle.available_moves    |                              |
```

---

## PokemonBattleEnv

Our environment extends `poke-env`'s `SinglesEnv`:

```python
class PokemonBattleEnv(SinglesEnv):
    def __init__(self, reward_config=None, **kwargs):
        # SinglesEnv handles Showdown connection
        # We add: reward tracking, action mapping, observation formatting
```

### The Gymnasium Interface

```python
# Observation space: what the agent sees
observation_space = {
    "obs":          Box(-1.0, 10.0, (13, 164)),  # 13 tokens, 164 features each
    "species":      Box(0, VOCAB, (13,)),         # Species IDs for embeddings
    "items":        Box(0, VOCAB, (13,)),         # Item IDs for embeddings
    "abilities":    Box(0, VOCAB, (13,)),         # Ability IDs for embeddings
    "action_mask":  Box(0, 1, (14,)),             # Which actions are legal
}

# Action space: what the agent can do
action_space = Discrete(14)
# Actions 0-3:  Regular moves (4 move slots)
# Actions 4-7:  Gimmick moves (Dynamax, Mega, Z-Move, Tera)
# Actions 8-13: Switch to Pokemon 2-7 (6 bench slots)
```

### Action Space Design

```
Action Index   Meaning               When Valid
-----------    ------                ----------
0              Use Move 1            Always (if move exists and PP > 0)
1              Use Move 2            Always (if move exists and PP > 0)
2              Use Move 3            Always (if move exists and PP > 0)
3              Use Move 4            Always (if move exists and PP > 0)
4              Use Move 1 (Gimmick)  Only if gimmick available
5              Use Move 2 (Gimmick)  Only if gimmick available
6              Use Move 3 (Gimmick)  Only if gimmick available
7              Use Move 4 (Gimmick)  Only if gimmick available
8              Switch to Slot 2      Only if Pokemon alive and not active
9              Switch to Slot 3      Only if Pokemon alive and not active
10             Switch to Slot 4      Only if Pokemon alive and not active
11             Switch to Slot 5      Only if Pokemon alive and not active
12             Switch to Slot 6      Only if Pokemon alive and not active
13             Switch to Slot 7      Only if Pokemon alive and not active
```

### The `calc_reward` Method

Called every step (every turn of the battle). Computes the delta reward:

```python
def calc_reward(self, battle):
    return self._compute_configured_delta_reward(battle)

def _compute_configured_delta_reward(self, battle):
    # Compare current state to previous state
    reward = 0.0

    # 1. HP change (how much HP changed this turn)
    #    Positive = we dealt damage, negative = we took damage
    reward += (our_hp_fraction - prev_our_hp) * hp_value_weight
    reward -= (opp_hp_fraction - prev_opp_hp) * hp_value_weight

    # 2. Fainting events
    for each newly fainted opponent: reward += fainted_value
    for each newly fainted ally:     reward -= fainted_value

    # 3. Type matchup bonus (how good our moves are against opponent)
    reward += matchup_quality * matchup_reward_weight

    # 4. Action quality (did we pick a good move?)
    reward += action_quality * action_quality_weight

    # 5. Terminal reward
    if won:  reward += victory_reward
    if lost: reward += defeat_penalty

    # 6. Scale everything down for value function stability
    return reward * reward_scale
```

> **WHY delta rewards?** Instead of giving absolute rewards ("you have 60% HP"), we give delta rewards ("you gained 10% HP relative to last turn"). This creates a much cleaner learning signal because the agent only needs to understand "did this turn help or hurt?" rather than "how good is my overall position?"

---

## The Action Mask

Not all 14 actions are legal every turn. If your Charizard only has 3 moves, action 3 is invalid. If all your other Pokemon fainted, actions 8-13 are invalid.

The action mask is a binary vector of shape (14,):
```
[1, 1, 1, 0, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0]
 |  |  |  |  |  |  |  |  |  |  |  |  |  |
 M1 M2 M3 M4 G1 G2 G3 G4 S2 S3 S4 S5 S6 S7

M = regular move, G = gimmick move, S = switch
1 = legal, 0 = illegal
```

In the model, invalid actions get their logits set to -1e8 before softmax, making their probability effectively zero. This is called "action masking."

> **WHY**: Without masking, the agent would need to learn which actions are legal through trial and error. With masking, it never wastes a turn on an illegal move. This is a huge advantage in a 14-action space where typically only 4-6 are legal.

---

## CurriculumSingleAgentWrapper

This wraps `PokemonBattleEnv` and adds curriculum learning support:

```python
class CurriculumSingleAgentWrapper:
    def __init__(self, env, opponent_mix, ...):
        self.env = env
        self.opponent_pool = {}  # Lazy opponent creation
        self.current_opponent = None

    def reset(self):
        # 1. Sample an opponent from the mix
        # e.g., {"random": 0.4, "random_no_switch": 0.3, "self": 0.3}
        opponent_type = random.choices(list(mix.keys()), weights=list(mix.values()))[0]

        # 2. Get or create the opponent
        opponent = self._get_or_create_opponent(opponent_type)

        # 3. Set it as the env's opponent
        self.env.set_opponent(opponent)

        # 4. Reset and return observation
        return self.env.reset()
```

### Opponent Types

| Type | Class | Behavior |
|------|-------|----------|
| `random` | `RandomPlayer` | Picks uniformly from ALL legal orders (moves AND switches) |
| `random_no_switch` | `RandomNoSwitchPlayer` | Picks uniformly from moves only. Switches only when forced. |
| `heuristic` | `SimpleHeuristicsPlayer` | Rule-based: picks super-effective moves, switches for type advantage |
| `self` | `SelfPlayPlayer` | Uses our own model weights. Temperature sampling (tau=0.8). |

> **WHY `random_no_switch` exists**: A purely random player wastes turns switching randomly, which inflates win rates. `random_no_switch` is a harder baseline because it always attacks, giving a more honest signal of the agent's move-selection quality.

### The Wrapper Hierarchy

```
Ray Worker
    |
    v
CurriculumSingleAgentWrapper
    | - Samples opponents from mix
    | - Tracks episode stats
    | - Collects self-play diagnostics
    |
    v
PokemonBattleEnv
    | - Manages Showdown connection
    | - Computes observations and rewards
    | - Converts actions to Showdown orders
    |
    v
poke-env SinglesEnv
    | - WebSocket connection to Showdown
    | - Battle state parsing
    | - Move/switch submission
    |
    v
Pokemon Showdown Server
```

---

## The Order-to-Action Conversion

Pokemon Showdown expects `BattleOrder` objects (like `BattleOrder(move="Flamethrower")` or `BattleOrder(switch="Charizard")`). Our agent outputs integer action indices (0-13). The conversion:

```python
def order_to_action(order, battle, fake=False, strict=True):
    """Convert a BattleOrder to an action index."""
    # Try strict conversion first
    # If that fails, fall back to random legal order
    # If that fails, return -2 (default action)
```

And the reverse (action index to BattleOrder):

```python
def action_to_order(action, battle):
    """Convert action index to BattleOrder for Showdown."""
    if action < 4:   # Regular move
        move = battle.available_moves[action]
        return BattleOrder(move)
    elif action < 8:  # Gimmick move
        move = battle.available_moves[action - 4]
        return BattleOrder(move, mega=True/terastallize=True/dynamax=True)
    else:             # Switch
        pokemon = battle.available_switches[action - 8]
        return BattleOrder(pokemon)
```

### Fallback System

The conversion has multiple fallback layers because Showdown is finicky:
1. **Strict**: Map action index directly to order
2. **Retry**: If strict fails, try random legal orders (up to 3 retries)
3. **Hard fallback**: Pick the first action that converts legally
4. **Default**: Return -2 (environment handles this gracefully)

> **WHY**: Showdown battles can have edge cases - moves get disabled, Pokemon get trapped, items affect move availability. The fallback system ensures the environment never crashes, even when the agent tries an action that became illegal between turns.

---

## Battle Lifecycle

```
1. env.reset()
   | - Start new battle on Showdown
   | - Sample opponent from curriculum mix
   | - Wait for battle to begin
   | - Return initial observation
   v
2. env.step(action)
   | - Convert action to BattleOrder
   | - Submit to Showdown
   | - Opponent makes their move
   | - Showdown resolves the turn
   | - Compute delta reward
   | - Compute new observation
   | - Check if battle ended
   v
3. If battle ended:
   | - Return (obs, final_reward, done=True, truncated=False, info)
   | - Worker starts a new battle
   v
4. If battle continues:
   | - Return (obs, reward, done=False, truncated=False, info)
   | - Go back to step 2
```

### What Triggers a "Step"

A "step" in the environment corresponds to one turn in the Pokemon battle:
1. Our agent selects an action (move or switch)
2. The opponent selects their action
3. Showdown resolves both actions simultaneously
4. We observe the new battle state
5. We compute the reward for this turn

---

## What's Next

You now understand how battles become environment steps. Chapter 05 dives into the observation pipeline - the specific features that make up the 13 tokens and their 164 dense dimensions plus categorical embeddings.
