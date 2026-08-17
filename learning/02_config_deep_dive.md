# Chapter 02: The Configuration System

## How `--preset optimal` Becomes a Complete Training Setup

The config system is a hierarchy of dataclasses. Each one owns a specific concern.
Understanding this hierarchy is essential - every other component reads from it.

---

## The Preset System

```python
def get_config(preset: str) -> TrainingConfig:
```

Five presets exist, each returning a `TrainingConfig` with different resource/quality tradeoffs:

| Preset | Purpose | Model Dim | Workers | Batch Size |
|--------|---------|-----------|---------|------------|
| `quick` | Fast testing | 128 | 2 | 1024 |
| `standard` | Default dev | 512 | 8 | 4096 |
| `memory_safe` | Low RAM | 256 | 4 | 2048 |
| `optimal` | RTX 5090 | 512 | 24 | 4096 |
| `large` | Maximum | 768 | 32 | 8192 |

The presets differ in `num_workers`, `num_envs_per_worker`, model dimensions, batch sizes, and learning rate. The core algorithm (PPO) and reward structure stay the same.

---

## The Dataclass Hierarchy

```
TrainingConfig
  |
  +-- ModelConfig            # Neural network architecture
  +-- PPOConfig              # PPO algorithm hyperparameters
  +-- EnvironmentConfig      # Showdown servers, battle format, parallelism
  +-- RewardConfig           # Reward function weights and scaling
  +-- CurriculumConfig       # Progressive difficulty stages
  |     +-- CurriculumStageConfig[]  # Individual stages
  |           +-- RewardConfig       # Per-stage reward overrides
  +-- ValidationScheduleConfig  # When and how to evaluate
```

Each sub-config is a `@dataclass` with typed fields and default values.

---

## ModelConfig - The Neural Network

```python
@dataclass
class ModelConfig:
    # Token structure
    num_tokens: int = 13          # 1 global + 6 our team + 6 opponent team
    token_dim: int = 164          # Features per token (HP, types, moves, etc.)

    # Embedding dimensions
    species_embed_dim: int = 32   # Pokemon species embedding size
    item_embed_dim: int = 32      # Item embedding size
    ability_embed_dim: int = 32   # Ability embedding size
    # Total categorical: 32 * 3 = 96 dims
    # Total per-token input: 164 (dense) + 96 (categorical) = 260 dims

    # Transformer
    hidden_dim: int = 512         # Model hidden dimension
    num_heads: int = 8            # Attention heads (512/8 = 64 per head)
    num_transformer_layers: int = 2  # Transformer layers
    ffn_dim: int = None           # Feedforward dim (defaults to 4x hidden = 2048)

    # LSTM temporal memory
    use_lstm: bool = True         # Cross-turn memory
    lstm_hidden: int = 512        # LSTM hidden state size

    # Regularization
    dropout: float = 0.0068       # Very low dropout
```

> **WHY**: The model needs to process a *set* of Pokemon (your team + opponent team) where order matters (active vs bench) but exact positions don't. A transformer is ideal because it can attend to relationships between any pair of Pokemon. The LSTM adds memory across turns - knowing that the opponent used Protect last turn helps predict what they'll do this turn.

### Dimension Flow Through the Model

```
Input: (batch, 13 tokens, 260 dims)    # 164 dense + 96 categorical
         |
         v
Projection: 260 -> 512                 # Linear layer
         |
         v
+ Position/Role Embeddings: 512 -> 512  # Adds structural info
         |
         v
Transformer x2: 512 -> 512             # Self-attention between all tokens
         |
         v
CLS token extracted: (batch, 512)      # Token 0 summarizes the battle
         |
         +---> Policy Head: 512 -> 256 -> GELU -> 14 actions
         +---> Value Head:  512 -> 256 -> GELU -> 1 scalar
```

---

## PPOConfig - The Learning Algorithm

```python
@dataclass
class PPOConfig:
    # Learning
    lr: float = 0.0002                    # Learning rate (Adam optimizer)

    # Discounting
    gamma: float = 0.97                   # Discount factor (how much future rewards matter)
    lambda_: float = 0.87                 # GAE parameter (bias-variance tradeoff)

    # PPO clipping
    clip_param: float = 0.08              # Policy ratio clipping range [0.92, 1.08]

    # Exploration vs exploitation
    entropy_coeff: float = 0.013          # Bonus for diverse actions

    # Value function
    vf_loss_coeff: float = 0.5            # Value loss weight in total loss
    vf_clip_param: float = 4.85           # Value function clipping range

    # Optimization
    grad_clip: float = 5.0                # Max gradient norm
    train_batch_size: int = 4096          # Total timesteps collected per update
    sgd_minibatch_size: int = 512         # Minibatch size for SGD
    num_sgd_iter: int = 8                 # Epochs per update (reuses same batch)
```

> **WHY** (the journey of these values): These aren't standard defaults. They were tuned through months of debugging:
> - `grad_clip=5.0` (was 0.5): Original value clipped all gradients to 0.77% of their norm. The transformer/embedding layers got ~65 norm, so 0.5/65 = 0.77%. Effectively zero learning rate for those layers.
> - `gamma=0.97` (was 0.99 → 0.95 → 0.97): 0.99 gave too long a horizon (~100 steps for a 25-turn battle). 0.95 was too short. 0.97 is the Goldilocks value.
> - `entropy_coeff=0.013` (was 0.005 → 0.05 → 0.2 → 0.013): 0.005 was negligible. 0.2 caused too much randomness. Optuna sweep found 0.013 as optimal.
> - `clip_param=0.08` (standard is 0.2): Tighter clipping prevents the policy from changing too fast in this complex environment.

> See **Chapter 07** for the full PPO theory.

---

## EnvironmentConfig - The Battle Setup

```python
@dataclass
class EnvironmentConfig:
    # Battle format
    battle_format: str = "gen8randombattlenogimmicks"  # BDSP, no Dynamax

    # Fixed team (optional)
    player_team_path: str = "data/teams/player_team.txt"  # Our team file
    # When set, auto-converts format: *randombattle -> *customgame

    # Showdown servers
    showdown_host: str = "localhost"
    num_servers: int = 8          # Parallel Showdown instances
    start_port: int = 8000        # Ports 8000-8007

    # Parallelism
    num_workers: int = 24         # Ray rollout workers
    num_envs_per_worker: int = 8  # Environments per worker (24 * 8 = 192 parallel battles)
    batch_mode: str = "truncate_episodes"  # How to handle episode boundaries

    # Battle constraints
    max_steps_per_battle: int = 200  # Max turns before truncation
```

### The Server Architecture

```
                    Trainer Process
                         |
                    Ray (distributed)
                    /    |    \       \
              Worker1 Worker2 ... Worker24
              / | \   / | \     / | \
            E1 E2 E3 E4 E5 E6 ... E7 E8  (8 envs per worker)
             |  |  |  |  |  |     |  |
            :8000    :8001 :8002 ... :8007
             |        |     |        |
          Showdown  Showdown Showdown Showdown  (8 server instances)
```

Each environment connects to one of the 8 Showdown servers. With 24 workers x 8 envs = 192 parallel battles distributed across 8 servers. That's ~24 battles per server simultaneously.

> **WHY 8 servers**: Pokemon Showdown is single-threaded Node.js. A single instance can't handle 192 concurrent battles without becoming a bottleneck. 8 servers spread the load. The number was chosen to match `num_servers` in the spin-up script.

---

## RewardConfig - The Reward Function

```python
@dataclass
class RewardConfig:
    # Terminal rewards (end of battle)
    victory_reward: float = 10.0         # Win
    defeat_penalty: float = -10.0        # Lose

    # Per-step shaping rewards
    hp_value_weight: float = 2.0         # HP advantage signal
    fainted_value: float = 5.0           # Pokemon KO'd signal
    matchup_reward_weight: float = 0.2   # Type effectiveness bonus
    action_quality_weight: float = 0.3   # Good move selection bonus

    # Global scaling
    reward_scale: float = 0.05           # Scales all rewards (critical for value function)
```

### How Rewards Combine

For a single step, the reward is:

```
delta_hp_reward = (our_hp_change - opp_hp_change) * hp_value_weight
faint_reward    = (opponent_fainted * fainted_value - our_fainted * fainted_value)
matchup_reward  = type_effectiveness_score * matchup_reward_weight
action_reward   = move_quality_score * action_quality_weight
terminal_reward = +10 if won, -10 if lost, 0 if ongoing

total_reward = (delta_hp_reward + faint_reward + matchup_reward + action_reward + terminal_reward) * reward_scale
```

> **WHY `reward_scale=0.05`**: The value function was completely unstable without this. Raw rewards are in the range [-15, +15]. RLlib normalizes advantages but NOT returns. The value function was chasing a huge, unscaled target. Scaling to [-0.75, +0.75] made explained variance jump from 0.1 to 0.7.

> See **Chapter 12** for the full reward engineering history.

---

## CurriculumConfig - Progressive Difficulty

```python
@dataclass
class CurriculumConfig:
    enabled: bool = True
    stages: List[CurriculumStageConfig] = ...  # Ordered list of stages

    # Promotion mechanics
    rolling_window_episodes: int = 300   # Window for win rate calculation
    min_episodes_before_promotion: int = 3000  # Minimum episodes before advancing
```

### CurriculumStageConfig

```python
@dataclass
class CurriculumStageConfig:
    name: str                              # e.g., "moves_and_switches_warmup"
    promote_at_win_rate: float             # Win rate threshold (e.g., 0.65 = 65%)
    opponent_mix: Dict[str, float]         # {"random": 0.4, "self": 0.3, ...}
    reward_config: RewardConfig            # Stage-specific reward overrides
```

### Current Stages

```
Stage 0: "moves_and_switches_warmup"
  - Opponents: 40% random, 30% random_no_switch, 30% self
  - Promote at: 65% win rate (rolling 300 episodes)
  - Minimum: 3000 episodes before promotion possible

Stage 1: "mixed_final"
  - Opponents: 50% heuristic, 40% self, 10% random_no_switch
  - Promote at: >1.0 (never promotes - this is the terminal stage)
```

> See **Chapter 08** for the full curriculum design.

---

## ValidationScheduleConfig - Evaluation

```python
@dataclass
class ValidationScheduleConfig:
    enabled: bool = True
    protocols: List[str] = ["benchmark"]  # Which validation protocols to run
    validation_freq_steps: int = 100_000  # Run validation every N training steps

    # Benchmark-specific
    benchmark_episodes_per_opponent: int = 50  # Episodes per opponent tier
    benchmark_opponents: List[str] = ["random", "random_no_switch", "heuristic"]
    max_steps_per_battle: int = 200
```

A benchmark validation runs 3 opponents x 50 episodes = 150 battles, measuring win rate against each tier.

> See **Chapter 10** for the full validation system.

---

## How Config Flows Through the System

```
train_battler.py
    |
    v
get_config("optimal") -> TrainingConfig
    |
    v
PokemonTrainer(config)
    |
    +-- config.model -> rllib_config_builder -> PPO algorithm
    +-- config.ppo -> rllib_config_builder -> PPO hyperparameters
    +-- config.environment -> rllib_config_builder -> env registration
    +-- config.reward -> battle_env -> reward computation
    +-- config.curriculum -> CurriculumManager -> stage progression
    +-- config.validation -> _run_scheduled_validation -> evaluation
```

Every component receives only the config slice it needs. The trainer orchestrates the handoff.

---

## Key Insight: Per-Stage Reward Overrides

Each `CurriculumStageConfig` has its own `reward_config`. This means the reward function can change as the agent progresses. For example, early stages might have higher `matchup_reward_weight` to teach type effectiveness, while later stages reduce it so the agent focuses on strategy.

This is why `rllib_config_builder.py` uses `stage.reward_config` (not `config.reward`) when the curriculum is active. Getting this wrong silently ignores reward parameters - a bug that wasted an entire hyperparameter sweep.

---

## What's Next

Now you understand the configuration. Chapter 03 follows the training loop - what happens inside `trainer.train()`.
