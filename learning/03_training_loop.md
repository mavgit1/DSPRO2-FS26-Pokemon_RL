# Chapter 03: The Training Loop

## Inside `PokemonTrainer.train()`

This is the heart of the project. The training loop runs millions of steps,
collecting battle experience and updating the neural network.

---

## The Big Picture

```
trainer.train()
    |
    v
PHASE 1: SETUP
    ray.init()              # Start distributed runtime
    register_environments() # Tell Ray about our Gymnasium env
    build_ppo_config()      # Configure PPO algorithm
    PPO(config)             # Create algorithm + model (allocated to GPU)
    [restore checkpoint]    # If resuming
    mlflow.start_run()      # Begin experiment tracking
    |
    v
PHASE 2: TRAINING LOOP (repeats until total_timesteps)
    result = algo.train()   # THE CORE STEP (see below)
    collect_metrics(result) # Extract and log metrics
    export_selfplay_weights() # Save model for self-play opponent
    update_curriculum()     # Check if agent should advance difficulty
    run_validation()        # Periodic evaluation (every 100k steps)
    save_checkpoint()       # Periodic save (every 150k steps)
    |
    v
PHASE 3: CLEANUP
    save_final_model()
    log_summary()
    ray.shutdown()
```

---

## Phase 1: Setup (Happens Once)

### 1a. Ray Initialization

```python
ray.init(
    num_gpus=config.num_gpus,  # 1 for optimal preset
    # Ray manages a cluster of workers on your machine
)
```

> **CONCEPT: What is Ray?**
> Ray is a distributed computing framework. Think of it as a process manager that can run Python functions on multiple CPU cores (and GPUs) simultaneously. In our case:
> - **Rollout workers**: Separate processes that each run multiple battle environments. They collect experience (observations, actions, rewards) in parallel.
> - **Learner**: The process that takes all that collected experience and updates the neural network on the GPU.
>
> Without Ray, you'd run one battle at a time. With Ray, you run 192 battles in parallel.

### 1b. Environment Registration

```python
register_environments(config)  # In rllib_config_builder.py
```

This tells Ray: "Hey, I have a custom Gymnasium environment called `pokemon_battle_v0`. Here's how to create it." Each worker will create its own instances of this environment.

Key detail: The registration function maps environments to Showdown server ports. Worker 0's envs use port 8000, Worker 1's use 8001, etc. This distributes load evenly.

### 1c. PPO Algorithm Creation

```python
algo = PPO(config=ppo_config)  # The RLlib PPO algorithm
```

This single line creates:
- The neural network model (BattleTransformer)
- The policy (how to select actions)
- The replay buffer (stores collected experience)
- The optimizer (Adam, with all the PPO hyperparameters)
- The rollout worker pool (24 workers x 8 envs each)

The model is placed on the GPU. Workers run on CPU cores.

### 1d. Checkpoint Restoration (Optional)

```python
if resume_checkpoint:
    algo.restore(resume_checkpoint)
```

Restores model weights, optimizer state, and training step counter. The agent continues exactly where it left off.

---

## Phase 2: The Core Training Loop

```python
for iteration in range(max_iterations):
    result = algo.train()  # <-- This is the expensive part
    ...
```

### What `algo.train()` Does (Expanded)

Each call to `algo.train()` performs one full PPO update cycle:

```
algo.train()
    |
    v
STEP 1: COLLECT EXPERIENCE (Rollout Phase)
    |
    |   Each of 24 workers runs 8 environments in parallel.
    |   Each environment plays Pokemon battles on Showdown.
    |   For every battle turn:
    |     - Environment sends observation to policy
    |     - Policy returns action (with some randomness for exploration)
    |     - Environment executes action on Showdown
    |     - Showdown returns new battle state
    |     - Environment computes reward
    |   Continue until train_batch_size (4096) steps collected total.
    |
    v
STEP 2: COMPUTE ADVANTAGES (GAE Phase)
    |
    |   For each collected trajectory:
    |     - Use value function to estimate state values
    |     - Compute GAE (Generalized Advantage Estimation)
    |     - Advantages tell us: "Was this action better or worse than expected?"
    |
    v
STEP 3: UPDATE POLICY (Learning Phase)
    |
    |   Split the 4096 steps into 8 minibatches of 512.
    |   For each minibatch, for 8 epochs:
    |     - Forward pass through model
    |     - Compute PPO clipped loss (policy)
    |     - Compute value function loss
    |     - Compute entropy bonus
    |     - Total loss = policy_loss + vf_coeff * vf_loss - entropy_coeff * entropy
    |     - Backward pass
    |     - Clip gradients (max norm = 5.0)
    |     - Update weights
    |
    v
STEP 4: RETURN METRICS
    - policy_loss, vf_loss, entropy
    - episode_reward_mean, episode_len_mean
    - num_env_steps_trained (total steps so far)
```

> **CONCEPT: Why 4096 steps then 8 minibatches x 8 epochs?**
> PPO is an "on-policy" algorithm - it can only learn from experience collected by the current policy. Once it updates the policy, old experience is stale. So it collects a fixed batch (4096 steps), then squeezes as much learning as possible from that batch (8 epochs x 8 minibatches = 64 gradient updates) before throwing it away and collecting fresh experience.

---

## What Happens After `algo.train()` Returns

### Collecting Metrics

```python
metrics = collect_metrics(result)  # From src/training/metrics/
```

This extracts dozens of metrics from the RLlib result dict:
- **PPO metrics**: policy_loss, vf_loss, entropy, kl_divergence, clip_fraction
- **Episode metrics**: reward_mean, length_mean, win_rate
- **Runtime metrics**: steps_per_second, GPU utilization
- **System metrics**: CPU/RAM usage

All metrics are logged to MLflow.

### Self-Play Weight Export

```python
self._export_selfplay_weights()
```

Every iteration **that uses self/historical**, the current model weights are saved to `checkpoints/selfplay_latest.pt`. Self-play opponents load that file once per episode (`begin_episode`), not every turn.

> **CRITICAL**: The path must be absolute. Ray workers run from `/tmp/ray/...` and can't resolve relative paths. This was the root cause of a month-long bug where self-play opponents played randomly because weights never loaded.

### Curriculum Update

```python
self._update_curriculum(outcomes)
```

1. Collect recent battle outcomes from all workers (win/loss)
2. Feed them to the `CurriculumManager`
3. If win rate exceeds the promotion threshold:
   - Advance to the next stage
   - Push new opponent mix and reward config to all workers
4. Log curriculum metrics to MLflow

### Scheduled Validation

```python
if self.should_validate():
    self._run_scheduled_validation()
```

Every `validation_freq_steps` (default 100,000), run a full benchmark evaluation:
- 3 opponents x 50 episodes = 150 battles
- Compute win rates with Wilson confidence intervals
- Calculate composite skill score
- Print results to console
- Log to MLflow

> See **Chapter 10** for the validation system.

### Checkpoint Saving

```python
if self.should_checkpoint():
    self._save_checkpoint()
```

Every `checkpoint_freq` (default 150,000) steps:
- Save full algorithm state (model, optimizer, replay buffer)
- Save to `checkpoints/step_XXXXXX/`
- Can be restored later with `--resume-checkpoint`

---

## The Complete Iteration Timeline

```
Iteration N:
  |
  [Collect 4096 steps from 192 parallel battles]  ~30-60 seconds
  |
  [PPO update: 8 minibatches x 8 epochs]           ~5-10 seconds (GPU)
  |
  [Export self-play weights]                         ~0.1 seconds
  |
  [Collect metrics, log to MLflow]                   ~0.5 seconds
  |
  [Update curriculum]                                ~0.1 seconds
  |
  [Validation? (every 100k steps)]                   ~120 seconds
  |   150 battles against random/random_no_switch/heuristic
  |   In-process: no subprocess overhead
  |
  [Checkpoint? (every 150k steps)]                   ~10 seconds
  |
  [Diagnostics logging]                              ~0.1 seconds
  |
  Total per iteration: ~35-70 seconds (without validation)
```

---

## How the Training Loop Ends

```python
# After total_timesteps reached:
algo.save("final_model")
mlflow.end_run()
ray.shutdown()
```

The final model is saved. MLflow run is closed. Ray workers are terminated.

---

## Understanding the Metrics

During training, you'll see metrics logged. Here's what matters:

| Metric | Good Range | Bad Sign |
|--------|-----------|----------|
| `episode_reward_mean` | Increasing over time | Stuck or decreasing |
| `policy_loss` | ~0.01-0.1 (should stabilize) | Very high or diverging |
| `vf_loss` | ~0.08-0.15 (should converge) | Very high or increasing |
| `entropy` | 1.0-2.5 (gradually decreasing) | Drops below 0.5 too fast |
| `kl_divergence` | < 0.02 (policy not changing too fast) | > 0.05 (updates too aggressive) |
| `explained_variance` | > 0.5 (value function learning) | < 0.2 (value function useless) |
| `win_rate` | Increasing per stage | Stuck below 50% |

### Metric Flow Diagram

```
algo.train()
    |
    v
result dict (from RLlib)
    |
    v
collect_metrics() in src/training/metrics/
    |
    +---> PPO metrics (loss, entropy, KL, clip_frac)
    +---> Episode metrics (reward, length, win_rate)
    +---> Runtime metrics (steps/sec, throughput)
    +---> System metrics (CPU, RAM, GPU)
    |
    v
log to MLflow (every iteration)
log to console (every iteration)
log to self-play diagnostics (every iteration)
```

---

## What's Next

Now you understand the training loop at a high level. Chapter 04 dives into the environment - how a Pokemon battle on Showdown becomes the observations and rewards that feed this loop.
