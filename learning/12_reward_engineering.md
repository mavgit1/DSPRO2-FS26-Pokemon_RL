# Chapter 12: Reward Engineering

## The Full History of Getting the Reward Right

Reward engineering was the most painful part of this project.
The reward function was rewritten multiple times, with each version
revealing new problems. This chapter traces the journey.

---

## CONCEPT: The Reward Hypothesis

The RL hypothesis states: any goal can be described as maximizing cumulative reward. But designing the right reward function is extremely hard.

**Good reward**: Winning the battle. Clear, unambiguous.
**Problem**: Winning only happens at the end. With ~20 turns per battle, the agent needs to learn which of 20 turns contributed to the win. This is the **credit assignment problem**.

**Reward shaping** adds per-step signals to help the agent learn faster:
- "Your HP went up relative to the opponent" (positive signal)
- "You knocked out a Pokemon" (positive signal)
- "You used a super-effective move" (positive signal)

But bad shaping can be worse than no shaping.

---

## Version 1: Terminal Rewards Only

```python
reward = +10 if won else -10 if lost else 0
```

**Result**: Agent didn't learn. The +10 or -10 signal was 20 turns away from most actions. No gradient signal for individual moves.

**Lesson**: Pure terminal rewards are too sparse for a 20-turn battle.

---

## Version 2: Heavy Shaping

```python
reward += matchup_quality * 5.0    # Type effectiveness
reward += action_quality * 2.0     # Move selection quality
```

**Problem: Reward hacking.** The agent learned to maximize type effectiveness scores without actually winning battles. It would pick "super-effective" moves that dealt minimal damage over moves that would actually win.

**Lesson**: Shaping rewards that are too strong create local optima. The agent optimizes the proxy instead of the true objective (winning).

---

## Version 3: No Shaping (Overcorrection)

```python
reward = delta_hp_signal * 1.0
# Removed matchup_reward_weight and action_quality_weight entirely
```

**Problem: Gradient cancellation.** Without per-step action signals, the policy gradient had nothing to differentiate between good and bad moves. All moves in a winning battle got equal positive gradient, and all moves in a losing battle got equal negative gradient. The net effect was near-zero learning.

**Lesson**: You need SOME per-step signal. The trick is keeping it small enough to guide but not dominate.

---

## Version 4: Balanced Shaping (Current)

```python
reward = 0.0

# Primary signal: HP changes (directly correlates with winning)
reward += (our_hp_delta - opp_hp_delta) * hp_value_weight  # 2.0

# Fainting events (strong discrete signal)
reward += opponent_fainted * fainted_value   # +5.0
reward -= our_fainted * fainted_value        # -5.0

# Weak shaping (guides but doesn't dominate)
reward += matchup_quality * matchup_reward_weight    # 0.2 (very small)
reward += action_quality * action_quality_weight     # 0.3 (very small)

# Terminal reward (the actual objective)
reward += victory_reward if won else defeat_penalty  # +/-10

# Scale everything down for value function stability
reward *= reward_scale  # 0.05
```

### The Magnitude Hierarchy

```
Reward Component           Typical Value    Weighted Value    Scaled (x0.05)
-------------------------  --------------   ----------------  ---------------
Terminal victory           +10.0            +10.0             +0.50
Fainting opponent          +5.0             +5.0              +0.25
HP swing (big turn)        ~0.3             ~0.6              +0.03
Matchup quality            ~0.5             ~0.1              +0.005
Action quality             ~0.3             ~0.09             +0.0045
```

The terminal reward dominates. Shaping rewards are tiny in comparison. This ensures the agent optimizes for winning, with shaping only providing directional hints.

---

## The Reward Scale Fix

### The Problem

Even with balanced shaping, the value function's explained variance was stuck at ~0.1. It couldn't predict returns at all.

### Root Cause

Raw returns are in the range [-15, +15]. RLlib normalizes **advantages** but NOT **returns**. The value function target was a huge, unscaled number:

```
Value function trying to predict: returns in [-15, +15]
Value function output range:      ~[-2, +2] (typical neural network)
Result: predictions are always wrong, loss never converges
```

### The Fix

```python
reward_scale = 0.05  # Scales returns to [-0.75, +0.75]
```

### Results

```
Before reward_scale:
  Explained variance: ~0.1 (useless value function)
  Value loss: high, no downward trend

After reward_scale = 0.1:
  Explained variance: 0.53 - 0.87 (learning well)
  Value loss: 0.08 - 0.13 (converged)

After reward_scale = 0.05 (current):
  Even more stable
```

---

## The HP Value Weight Journey

```python
hp_value_weight = 2.0  # Current value
```

This was increased from 1.0 to 2.0 because:
- HP is the most direct signal of battle progress
- With weight 1.0, the HP signal was too weak compared to terminal rewards
- With weight 2.0, the agent can learn "dealing damage is good" within a few episodes

---

## Matchup Reward Computation

```python
def _compute_matchup_quality(battle):
    """Score type effectiveness of our best available move."""
    our_active = battle.active_pokemon
    opp_active = battle.opponent_active_pokemon

    best_effectiveness = 0.0
    for move in our_active.moves.values():
        effectiveness = move.type.damage_multiplier(
            *opp_active.types
        )
        best_effectiveness = max(best_effectiveness, effectiveness)

    # Map effectiveness to [-1.0, 1.0]
    if best_effectiveness >= 2.0: return 1.0    # Super effective
    if best_effectiveness >= 1.0: return 0.0    # Neutral
    if best_effectiveness >= 0.5: return -0.5   # Not very effective
    return -1.0                                 # Immune
```

> **WHY this exists**: Even at low weight (0.2), this signal helps the agent discover type matchups early in training. Without it, the agent needs thousands of episodes to learn "Fire beats Grass" through HP signals alone.

---

## Action Quality Computation

```python
def _compute_action_quality(battle):
    """Score the quality of the last action taken."""
    quality = 0.0

    # Offensive: Did we pick a good damaging move?
    if last_move_was_damaging:
        expected_power = best_available_power
        actual_power = last_move_power
        if actual_power >= expected_power * 0.8:
            quality += 0.5  # Picked a good move
        else:
            quality -= 0.5  # Picked a suboptimal move

    # Defensive: Does our active resist the opponent's best move?
    if our_active_resists_opponent:
        quality += 0.5  # Good positioning

    return quality
```

> **WHY**: This gives per-action feedback. "You used Flamethrower against a Grass type, good job!" or "You switched to a Pokemon weak to the opponent, that was risky." At weight 0.3, it nudges the agent without creating hacking incentives.

---

## The Step Penalty (Defined but NOT Applied)

```python
# In RewardConfig:
step_penalty: float = -0.005
```

This is defined in the config but NOT used in `_compute_configured_delta_reward`. It exists in a standalone `compute_reward()` function that isn't called during training.

> **WHY**: The idea was to discourage long battles (encourage efficient play). But it introduced problems - the agent would sometimes forfeit good positions to avoid the penalty. It was removed but the config field remains.

---

## Delta Rewards vs Absolute Rewards

### Delta (Our Approach)
```python
reward = (current_value - previous_value) * weight
```
- Measures *change* from last turn
- Zero-mean (positive and negative equally likely)
- Better for learning: "Was this turn better or worse?"

### Absolute (Not Used)
```python
reward = current_value * weight
```
- Measures *current state*
- Always positive (or always negative)
- Harder to learn from: "I have 60% HP" doesn't tell the agent what to do differently

> **WHY delta**: Delta rewards naturally create a comparison signal. "My HP went from 80% to 60%" is more actionable than "I have 60% HP."

---

## The Complete Reward Function

```python
def _compute_configured_delta_reward(battle):
    reward = 0.0

    # HP signal (delta)
    our_hp = sum(mon.current_hp_fraction for mon in our_team)
    opp_hp = sum(mon.current_hp_fraction for mon in opp_team)
    reward += (our_hp - prev_our_hp) * hp_value_weight      # +2.0 weight
    reward -= (opp_hp - prev_opp_hp) * hp_value_weight      # +2.0 weight

    # Fainting events (discrete)
    for newly_fainted_opp:  reward += fainted_value          # +5.0
    for newly_fainted_ours: reward -= fainted_value          # -5.0

    # Weak shaping
    reward += _compute_matchup_quality(battle) * 0.2
    reward += _compute_action_quality(battle) * 0.3

    # Terminal
    if won:  reward += 10.0
    if lost: reward -= 10.0

    # Scale for value function stability
    return reward * 0.05
```

### Typical Battle Reward Trace

```
Turn 1:  +0.04  (dealt some damage, neutral matchup)
Turn 2:  +0.07  (dealt more damage, good matchup)
Turn 3:  +0.15  (fainted opponent's Pokemon)
Turn 4:  -0.03  (opponent switched to counter)
Turn 5:  +0.02  (neutral exchange)
...
Turn 20: +0.50  (won the battle!)
Total:   ~+1.5  (scaled)
```

---

## What's Next

Chapter 13 summarizes every major design decision in one place.
