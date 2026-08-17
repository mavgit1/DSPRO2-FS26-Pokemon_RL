# Chapter 08: Curriculum Learning

## From Easy Opponents to Hard Ones

Training an RL agent against the hardest opponent from day one doesn't work.
The agent loses every battle and never learns anything. Curriculum learning
solves this by gradually increasing difficulty.

---

## CONCEPT: Why Curriculum Learning?

**Problem**: If the agent plays against `SimpleHeuristicsPlayer` (the hardest opponent) from the start:
- It loses 95%+ of battles
- Rewards are almost always negative
- No positive signal to learn from
- The agent never discovers basic strategies

**Solution**: Start with easy opponents, learn basics, then increase difficulty.

```
Difficulty Curve:
  |
  |                          * Stage 2: mixed_final
  |                        /
  |                     /
  |            * Stage 1: warmup
  |          /
  |        /
  | * Baseline: can barely beat random
  +--------------------------------------> Training progress
```

---

## Our Curriculum Design

### Stage 0: "moves_and_switches_warmup"

**Goal**: Learn basic move selection and when to switch.

```python
CurriculumStageConfig(
    name="moves_and_switches_warmup",
    promote_at_win_rate=0.65,     # Need 65% win rate to advance
    opponent_mix={
        "random": 0.4,            # 40% vs fully random
        "random_no_switch": 0.3,  # 30% vs random-no-switch
        "self": 0.3,              # 30% vs self-play
    },
    reward_config=RewardConfig(...),  # Specialized rewards for learning basics
)
```

**What the agent learns here:**
- "Using super-effective moves is good" (from matchup_reward)
- "Keeping HP high is good" (from hp_value_weight)
- "Fainting is bad" (from fainted_value)
- Basic type matchups (Fire > Grass, Water > Fire, etc.)

**Promotion requirement**: 65% win rate over the last 300 episodes, with at least 3000 episodes played.

### Stage 1: "mixed_final" (Terminal Stage)

**Goal**: Learn advanced strategy against strong opponents.

```python
CurriculumStageConfig(
    name="mixed_final",
    promote_at_win_rate=1.1,      # > 1.0 = never promotes (terminal)
    opponent_mix={
        "heuristic": 0.5,         # 50% vs rule-based player
        "self": 0.4,              # 40% vs self-play
        "random_no_switch": 0.1,  # 10% vs easy opponent (maintenance)
    },
    reward_config=RewardConfig(...),  # Full difficulty rewards
)
```

**What the agent learns here:**
- How to beat a rule-based player (much harder than random)
- Self-play dynamics (adapting to its own strategies)
- Advanced switching and prediction

> **WHY keep 10% random_no_switch in the final stage?** Purely removing easy opponents causes "catastrophic forgetting" - the agent might unlearn basic strategies. Keeping a small fraction of easy opponents acts as a maintenance signal.

---

## The Curriculum Manager

```python
class CurriculumManager:
    def __init__(self, config):
        self.current_stage_idx = 0
        self.outcome_window = deque(maxlen=rolling_window_episodes)
        self.episodes_in_stage = 0

    def update(self, new_outcomes):
        """Feed new battle results. Returns True if stage changed."""
        self.outcome_window.extend(new_outcomes)
        self.episodes_in_stage += len(new_outcomes)

        if self._can_promote():
            self.current_stage_idx += 1
            return True
        return False

    def _can_promote(self):
        stage = self.stages[self.current_stage_idx]
        # Need enough episodes
        if self.episodes_in_stage < min_episodes:
            return False
        # Need enough data in rolling window
        if len(self.outcome_window) < rolling_window_episodes / 2:
            return False
        # Win rate must exceed threshold
        win_rate = sum(self.outcome_window) / len(self.outcome_window)
        return win_rate >= stage.promote_at_win_rate
```

### Promotion Logic

```
Every training iteration:
  1. Collect battle outcomes from all workers
  2. Feed to CurriculumManager.update()
  3. If promoted:
     a. Log "Curriculum stage advanced" to MLflow
     b. Push new opponent mix to all workers
     c. Push new reward config to all workers
     d. Reset episode counter
```

---

## How Opponents Are Sampled

When the curriculum sets an opponent mix like `{"random": 0.4, "self": 0.3, "random_no_switch": 0.3}`, each worker samples independently for each new battle:

```python
def reset(self):
    # Sample from the mix
    opponent_type = random.choices(
        population=list(mix.keys()),
        weights=list(mix.values()),
        k=1
    )[0]

    # Get or create the opponent
    opponent = self._get_or_create_opponent(opponent_type)
    self.env.set_opponent(opponent)
```

### Opponent Pool

Opponents are created lazily and cached:

```python
class CurriculumSingleAgentWrapper:
    def _get_or_create_opponent(self, opponent_type):
        if opponent_type not in self.opponent_pool:
            if opponent_type == "random":
                self.opponent_pool[opponent_type] = RandomPlayer(...)
            elif opponent_type == "random_no_switch":
                self.opponent_pool[opponent_type] = RandomNoSwitchPlayer(...)
            elif opponent_type == "heuristic":
                self.opponent_pool[opponent_type] = SimpleHeuristicsPlayer(...)
            elif opponent_type == "self":
                self.opponent_pool[opponent_type] = SelfPlayPlayer(...)
        return self.opponent_pool[opponent_type]
```

> **WHY lazy creation?** Not every opponent type is needed in every stage. Creating all four upfront would waste resources on Showdown connections that never get used.

---

## Per-Stage Reward Overrides

Each stage can override the reward configuration:

```python
CurriculumStageConfig(
    name="warmup",
    reward_config=RewardConfig(
        matchup_reward_weight=0.4,   # Higher: really emphasize type matchups early
        action_quality_weight=0.5,   # Higher: reward good move selection
        ...
    ),
)

CurriculumStageConfig(
    name="mixed_final",
    reward_config=RewardConfig(
        matchup_reward_weight=0.1,   # Lower: agent knows type matchups by now
        action_quality_weight=0.2,   # Lower: focus on strategy, not individual moves
        ...
    ),
)
```

> **CRITICAL GOTCHA**: `rllib_config_builder.py` uses `stage.reward_config` (not `config.reward`) when curriculum is active. The hyperparameter sweep initially ignored this, meaning reward params from the sweep were silently ignored for months.

---

## Curriculum in the Env Bridge

When a stage changes, the trainer pushes the update to all workers:

```python
# In trainer.py
def _apply_curriculum_stage(self, stage):
    payload = {
        "opponent_mix": stage.opponent_mix,
        "reward_config": stage.reward_config,
    }
    apply_curriculum_stage(self.algo, payload)

# In env_bridge.py
def apply_curriculum_stage(algo, payload):
    """Push curriculum update to all environments on all workers."""
    foreach_env(algo, lambda env: env.apply_curriculum_stage(payload))
```

The `foreach_env` function traverses RLlib's wrapper hierarchy to reach every `CurriculumSingleAgentWrapper` instance across all workers and calls its `apply_curriculum_stage` method.

---

## Visual Timeline

```
Training Progress:
  |
  |  Stage 0: Warmup (random/random_no_switch/self)
  |  |
  |  | Episode 1-3000: Learning basics (win rate: 30% -> 50%)
  |  | Episode 3000-8000: Getting consistent (win rate: 50% -> 65%)
  |  | Episode 8000: PROMOTION (65% win rate achieved)
  |  |
  |  Stage 1: Mixed Final (heuristic/self/random_no_switch)
  |  |
  |  | Episode 8000-20000+: Learning advanced strategy
  |  | Win rate drops temporarily (harder opponents)
  |  | Gradually improves as agent adapts
  |  |
  |  (Never promotes - terminal stage)
  |
```

---

## What's Next

Chapter 09 covers self-play - how the agent plays against itself and the unique challenges that brings.
