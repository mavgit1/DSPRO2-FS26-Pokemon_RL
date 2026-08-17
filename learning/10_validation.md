# Chapter 10: Validation

## How We Measure Agent Quality

Training metrics (loss, entropy) tell us the model is learning.
Validation tells us it's actually getting better at Pokemon.

---

## CONCEPT: Why Separate Validation?

Training metrics can be misleading:
- **Loss decreasing** doesn't mean the agent is playing better
- **Reward increasing** could be reward hacking (exploiting the reward function)
- **Win rate during training** is against a changing opponent mix

Validation runs the agent against fixed opponents in a controlled setting,
giving an unbiased measure of battle quality.

---

## Validation Protocols

### Benchmark (Default)

The primary validation protocol. Runs 150 battles total:

```
3 opponents x 50 episodes = 150 battles
  |
  +---> random (50 eps):            Baseline. Any decent agent should win 90%+.
  +---> random_no_switch (50 eps):  Harder baseline. Tests move selection quality.
  +---> heuristic (50 eps):         Strong test. Tests strategic understanding.
```

**Output metrics:**
- Per-opponent win rate with 95% Wilson confidence interval
- Composite skill score (weighted average)
- Consistency (fraction of tiers with >50% win rate)

### Smoke Test

Quick 3-episode check against random. Used for debugging - just verifies the agent can complete a battle without crashing.

### Fixed Paired

40 episodes with pre-defined team matchups. Tests agent quality with specific team compositions.

### Mirror

40 episodes where both sides use the same team. Tests pure strategy skill (eliminates team advantage as a factor).

---

## CONCEPT: Wilson Confidence Interval

When you win 35 out of 50 battles, the true win rate isn't exactly 70%. It's somewhere in a range. The Wilson score interval gives us that range:

```
Wilson CI for 35/50 wins:
  Point estimate: 70%
  95% CI: [55.9%, 81.5%]
  "We're 95% confident the true win rate is between 55.9% and 81.5%"
```

### Why It Matters

With only 50 episodes, there's significant variance. Wilson CIs tell us:
- Is 70% vs random significantly better than 50%? (Yes, CI doesn't overlap 50%)
- Is 55% vs heuristic significantly better than random? (Maybe, CI overlaps 50%)

```python
def wilson_score_interval(wins, total, z=1.96):
    """95% confidence interval for binomial proportion."""
    n = total
    p_hat = wins / n
    denominator = 1 + z**2 / n
    center = (p_hat + z**2 / (2*n)) / denominator
    spread = z * sqrt((p_hat*(1-p_hat) + z**2/(4*n)) / n) / denominator
    return (center - spread, center + spread)
```

---

## Composite Metrics

### Skill Score

Weighted average of win rates across opponent tiers:

```python
weights = {
    "random": 1.0,           # Easy: base weight
    "random_no_switch": 1.5, # Medium: 50% bonus
    "heuristic": 2.0,        # Hard: 100% bonus
}

skill_score = sum(wr[tier] * weights[tier] for tier in tiers) / sum(weights.values())
```

A skill score of 0.5 means the agent wins about half its battles overall (accounting for difficulty).
A skill score of 0.8 is very good. A skill score of 1.0 means 100% against all tiers.

### Consistency

```python
consistency = sum(1 for tier in tiers if win_rate[tier] > 0.5) / len(tiers)
```

What fraction of opponent tiers have >50% win rate. Consistency = 1.0 means the agent beats everyone. Consistency = 0.33 means it only beats one of three tiers.

---

## In-Process vs Subprocess Validation

### In-Process (Default During Training)

```python
run_inprocess_validation(algo, config)
```

- Takes the live `algo` object directly
- No subprocess, no Ray init, no checkpoint restore
- Runs validation battles using the current model weights
- Fast: ~120 seconds for 150 battles
- Used automatically during training every 100k steps

### Subprocess (For Offline Evaluation)

```python
run_validation(checkpoint_path, config)
```

- Spawns a new Python process
- Initializes Ray, restores checkpoint, runs validation
- Slower: 30-60 seconds overhead + validation time
- Used by `scripts/validate_checkpoint.py` for evaluating saved checkpoints

---

## Parallel Benchmark Execution

The 150 benchmark battles are distributed across Showdown servers:

```python
def _run_benchmark(algo, config, num_servers=8):
    # Divide episodes across servers
    episodes_per_server = {
        port: episodes // num_servers
        for port in range(start_port, start_port + num_servers)
    }

    # Run in parallel using ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=num_servers) as executor:
        futures = []
        for port, count in episodes_per_server.items():
            future = executor.submit(
                _run_episodes_on_port,
                algo, count, port
            )
            futures.append(future)

        results = [f.result() for f in futures]

    return aggregate(results)
```

> **WHY ThreadPoolExecutor, not ProcessPoolExecutor?** Each thread creates a Gymnasium environment that connects to a different Showdown server. The GIL isn't a bottleneck because most time is spent waiting for Showdown responses (network I/O).

> **WHY chunk distribution, not round-robin?** Each environment runs its assigned episodes sequentially. Round-robin would require coordinating multiple environments to the same server simultaneously, which causes concurrent access issues.

---

## Explore Mode (Stochastic Policy)

```bash
uv run scripts/validate_checkpoint.py --checkpoint ... --explore
```

Normal validation uses **argmax** (always pick the best action). Explore mode uses **masked softmax sampling**:

```python
def _sample_masked_action(logits, mask, temperature=1.0):
    """Sample from softmax instead of taking argmax."""
    logits[~mask.bool()] = float('-inf')
    probs = F.softmax(logits / temperature, dim=-1)
    return torch.multinomial(probs, 1)
```

> **WHY**: Argmax evaluates the deterministic policy. But during training, the agent uses stochastic actions (for exploration). Explore mode evaluates the stochastic policy, which can reveal if the agent has a good probability distribution or just one good action.

---

## Validation Output

### Console Output

```
=== Benchmark Validation (Step 200,000) ===

vs random:            ████████████████████░ 96% [87-99%]  (48/50)
vs random_no_switch:  ██████████████████░░░ 82% [69-91%]  (41/50)
vs heuristic:         █████░░░░░░░░░░░░░░░ 22% [12-36%]  (11/50)

Skill Score:    0.60
Consistency:    0.67 (2/3 tiers >50%)
Overall Win Rate: 66.7% (100/150)
Avg Battle Length: 18.3 turns
```

### MLflow Metrics

```
benchmark/win_rate_vs_random:            0.96
benchmark/ci_lower_vs_random:            0.87
benchmark/ci_upper_vs_random:            0.99
benchmark/win_rate_vs_random_no_switch:  0.82
benchmark/ci_lower_vs_random_no_switch:  0.69
benchmark/ci_upper_vs_random_no_switch:  0.91
benchmark/win_rate_vs_heuristic:         0.22
benchmark/ci_lower_vs_heuristic:         0.12
benchmark/ci_upper_vs_heuristic:         0.36
benchmark/skill_score:                   0.60
benchmark/consistency:                   0.67
```

### JSON Report

A full JSON report is saved with every BattleResult:
- Per-episode: opponent, outcome, reward, steps, actions
- Aggregated: win rates, CIs, skill score, consistency

---

## Validation During Training

The trainer calls validation every `validation_freq_steps`:

```python
def _run_scheduled_validation(self):
    # 1. Run benchmark in-process (no subprocess)
    results = run_inprocess_validation(
        algo=self.algo,
        config=self.config,
        num_servers=self.num_servers,
    )

    # 2. Log to the ACTIVE MLflow run (not a nested run)
    for key, value in results.metrics.items():
        mlflow.log_metric(key, value, step=self.total_steps)

    # 3. Print console output
    print(format_validation_summary(results))
```

> **WHY no nested MLflow run?** Earlier versions created a nested run for each validation, which cluttered the experiment. Now validation metrics go directly into the training run with a `benchmark/` prefix.

---

## What's Next

Chapter 11 covers the distributed infrastructure - how Ray makes all of this run in parallel across dozens of workers.
