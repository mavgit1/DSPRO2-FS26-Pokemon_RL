# Chapter 07: PPO Explained

## The Learning Algorithm

This chapter explains Proximal Policy Optimization (PPO) from scratch,
then connects each concept to our specific implementation.

---

## CONCEPT: The RL Problem

In reinforcement learning, an **agent** interacts with an **environment**:

```
Agent                     Environment
  |                            |
  | action                     |
  |--------------------------->|
  |                            | new state
  |          obs, reward       |
  |<---------------------------|
  |                            |
  | action                     |
  |--------------------------->|
  |          obs, reward       |
  |<---------------------------|
  ...repeat until episode ends...
```

The agent's goal: maximize cumulative reward over time.

### Key Terms

| Term | Meaning | Pokemon Example |
|------|---------|----------------|
| State | Current situation | Battle state (HP, moves, etc.) |
| Action | What the agent does | Use Flamethrower, switch to Pikachu |
| Reward | Feedback signal | +0.5 for good turn, +10 for winning |
| Policy | Agent's decision rule | Neural network that picks actions |
| Value | Expected future reward | "This position is worth about 5 points" |
| Episode | One complete interaction | One Pokemon battle |
| Step | One action-observation cycle | One turn of a battle |

---

## CONCEPT: Policy Gradient Methods

The simplest idea: tweak the policy to make good actions more likely and bad actions less likely.

```
For each step:
  1. We took action A in state S and got reward R
  2. If R was high: increase probability of A in state S
  3. If R was low: decrease probability of A in state S
```

Mathematically:
```
loss = -log(probability(action)) * advantage
```

Where `advantage` = "how much better was this action than average?"

### The Problem with Vanilla Policy Gradient

It's unstable. One bad update can destroy the policy entirely. If the agent finds a good strategy and then a single update makes it 10x worse, there's no way back.

---

## CONCEPT: PPO - The Solution

PPO adds a **trust region**: the policy can only change a small amount per update. This prevents catastrophic updates while still allowing learning.

### The Clipped Objective

```python
# ratio = new_policy_prob / old_policy_prob
# ratio > 1: new policy likes this action MORE than old
# ratio < 1: new policy likes this action LESS than old

ratio = exp(new_log_prob - old_log_prob)

# Clipped ratio: restrict to [1 - clip_param, 1 + clip_param]
# With clip_param = 0.08: ratio in [0.92, 1.08]
clipped_ratio = clamp(ratio, 1 - 0.08, 1 + 0.08)

# The PPO loss takes the WORSE of clipped and unclipped
loss = -min(
    ratio * advantage,           # Unclipped
    clipped_ratio * advantage    # Clipped
)
```

**What this means in practice:**
- If advantage > 0 (good action): loss encourages higher probability, but max ratio = 1.08
- If advantage < 0 (bad action): loss encourages lower probability, but min ratio = 0.92
- The policy can only change ~8% per update for any given action

> **WHY clip_param = 0.08 (not the standard 0.2)?** Pokemon battles have a complex action space with 14 possible actions and sparse rewards. A larger clip (0.2) allowed too-rapid policy changes that destabilized learning. 0.08 keeps updates conservative.

---

## CONCEPT: The Value Function

The value function V(s) estimates: "If I play optimally from this state, what's my expected total reward?"

```
V(about_to_win)  = +8.5  (expect to win soon)
V(even_battle)   = +0.3  (slight advantage)
V(about_to_lose) = -7.2  (probably going to lose)
```

### Why We Need It

The value function is used to compute **advantages**:
```
advantage = reward + gamma * V(next_state) - V(current_state)
```

This tells us: "Was this action better or worse than the value function expected?" If the value function says this state is worth 5, and after taking action A the reward + next state value is 8, then advantage = 3 (action A was surprisingly good).

### Value Function Loss

```python
vf_loss = mean((V_predicted - V_target)^2)
```

Standard regression: minimize the squared error between predicted and actual returns.

> **WHY vf_clip_param = 4.85?** The value function loss is also clipped. Without clipping, a single bad prediction can push the value function far off. The clip restricts how much V can change per update. 4.85 was found through Optuna sweep.

---

## CONCEPT: GAE (Generalized Advantage Estimation)

Computing advantages from a single step is noisy. GAE averages advantages over multiple time steps:

```
GAE with lambda_ = 0.87:
  - 87% weight on 1-step advantage (low variance, some bias)
  - 13% weight on multi-step advantage (higher variance, less bias)
```

The `lambda_` parameter trades off bias vs variance:
- `lambda_ = 0`: Only use 1-step returns (low variance, high bias)
- `lambda_ = 1`: Use full returns (high variance, no bias)
- `lambda_ = 0.87`: A good middle ground

---

## CONCEPT: Entropy Bonus

PPO adds a bonus for policy entropy (randomness):

```python
entropy_bonus = entropy_coeff * mean(entropy(policy))
```

High entropy = the agent is still exploring many actions.
Low entropy = the agent has converged to a few actions.

> **WHY entropy_coeff = 0.013?**
> - 0.005: Too low. Policy collapsed to always picking the same move.
> - 0.2: Too high. Agent never committed to a strategy.
> - 0.05: Decent but still too random for complex battles.
> - 0.013: Optuna found this sweet spot. Enough exploration to discover good strategies, but not so much that it can't commit.

---

## How Our Hyperparameters Map to PPO

| Parameter | Value | PPO Concept | Effect |
|-----------|-------|-------------|--------|
| `lr` | 0.0002 | Adam learning rate | How fast weights change |
| `gamma` | 0.97 | Discount factor | How much future rewards matter |
| `lambda_` | 0.87 | GAE parameter | Bias-variance tradeoff in advantages |
| `clip_param` | 0.08 | PPO clipping | Max policy change per update |
| `entropy_coeff` | 0.013 | Entropy bonus | Exploration vs exploitation |
| `vf_loss_coeff` | 0.5 | Value loss weight | How much to prioritize value accuracy |
| `vf_clip_param` | 4.85 | Value clipping | Max value change per update |
| `grad_clip` | 5.0 | Gradient clipping | Prevents exploding gradients |
| `train_batch_size` | 4096 | Steps per update | How much experience per learning step |
| `sgd_minibatch_size` | 512 | Minibatch size | Splits batch for GPU memory |
| `num_sgd_iter` | 8 | PPO epochs | How many times to reuse the same batch |

---

## The Complete PPO Update (Pseudocode)

```python
def ppo_update(collected_experience):
    # collected_experience has: obs, actions, rewards, values, log_probs

    # 1. Compute returns and advantages using GAE
    advantages = compute_gae(rewards, values, gamma=0.97, lambda_=0.87)
    returns = advantages + values

    # 2. Normalize advantages (zero mean, unit variance)
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    # 3. Split into minibatches
    for epoch in range(8):  # num_sgd_iter
        for minibatch in split(4096, 512):
            obs, actions, old_log_probs, old_values, adv, ret = minibatch

            # 4. Forward pass
            new_log_probs, new_values, entropy = model(obs)

            # 5. Compute PPO clipped loss
            ratio = exp(new_log_probs - old_log_probs)
            clipped_ratio = clamp(ratio, 0.92, 1.08)
            policy_loss = -min(ratio * adv, clipped_ratio * adv).mean()

            # 6. Compute value function loss (clipped)
            vf_pred = new_values
            vf_clipped = old_values + clamp(vf_pred - old_values, -4.85, 4.85)
            vf_loss = max((vf_pred - ret)^2, (vf_clipped - ret)^2).mean()

            # 7. Compute entropy bonus
            entropy_bonus = -0.013 * entropy.mean()

            # 8. Total loss
            loss = policy_loss + 0.5 * vf_loss + entropy_bonus

            # 9. Backward pass
            loss.backward()

            # 10. Clip gradients
            clip_grad_norm_(model.parameters(), max_norm=5.0)

            # 11. Update weights
            optimizer.step()
            optimizer.zero_grad()
```

---

## The PPO Loss Landscape

```
Total Loss = Policy Loss + vf_coeff * Value Loss - entropy_coeff * Entropy
             (~0.02-0.1)      (~0.08-0.15)               (~1.0-2.5)

Actual magnitudes with our config:
Policy loss:     ~0.03
Value loss:      ~0.10  * 0.5  = 0.05
Entropy bonus:  -1.5    * 0.013 = -0.02
Total:           ~0.06
```

The entropy bonus is small compared to the other terms, which is intentional - it should encourage exploration without dominating the loss.

---

## Why PPO (and not DQN, SAC, or A2C)?

| Algorithm | Pros | Cons | Why Not For Us |
|-----------|------|------|----------------|
| PPO | Stable, well-tested, works with continuous+discrete, simple to implement | Sample inefficient | We chose it. Stability matters more than efficiency. |
| DQN | Simple, good for discrete actions | No policy gradient, needs replay buffer, struggles with stochastic policies | Can't easily handle action masking. No entropy bonus. |
| SAC | Sample efficient, good for continuous | Complex, overkill for discrete | Designed for continuous action spaces. |
| A2C | Simple | Less stable than PPO | PPO is essentially A2C with clipping. PPO is strictly better. |
| IMPALA | Very scalable | Extremely complex to implement | Overkill for our scale. |

PPO is the default choice for most RL projects. It's not the best at any one thing, but it's reliable and well-understood.

---

## What's Next

You now understand the learning algorithm. Chapter 08 covers curriculum learning - how we structure the training to go from easy opponents to hard ones.
