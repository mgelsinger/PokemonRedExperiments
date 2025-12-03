# PPO Hyperparameter Tuning Guide

## Current Default Hyperparameters

From `training/train_ppo.py`:

```python
DEFAULT_TRAIN_CONFIG = {
    "num_envs": 16,
    "total_multiplier": 50,
    "batch_size": 512,
    "n_epochs": 1,
    "gamma": 0.997,
    "ent_coef": 0.01,
    "learning_rate": 3e-4,
    "clip_range": 0.2,
    "vf_coef": 0.5,
    "max_grad_norm": 0.5,
    "gae_lambda": 0.95,
}
```

**Current Setting**: `rollout_horizon = 512` (from earlier memory optimization)

---

## Problem: Sparse Rewards in Pokemon

### Reward Sparsity Analysis

**Dense Signal** (every step):
- Exploration: ~0.1 per tile
- Step penalty: -0.001

**Sparse Signal** (every ~500 steps):
- Battle start: +10.0
- Battle win: +50.0

**Very Sparse Signal** (every ~10,000+ steps):
- Badge: +500.0
- Level up: +20.0

### Challenge for PPO

PPO uses advantage estimates to credit actions. With sparse rewards:
- Most timesteps have zero reward signal
- Advantage estimates rely heavily on value function
- Long credit assignment chains (action at t=0 → reward at t=500)

**Solution**: Tune hyperparameters for better credit assignment.

---

## Hyperparameter Breakdown

### 1. `gamma` (Discount Factor)

**What it does**: Determines how much future rewards matter.
```python
discounted_return = r_t + gamma * r_{t+1} + gamma^2 * r_{t+2} + ...
```

**Current**: 0.997 (very high - values rewards 500 steps away)

**Recommended for Sparse Rewards**:
- **Navigation**: 0.99 (shorter horizon)
- **Battles**: 0.995 (medium horizon)
- **Badge Quest**: 0.997 (long horizon needed)

**Reasoning**:
- Higher gamma = better long-term planning
- But higher gamma = slower learning (more variance)
- For battles that happen in 500 steps, `0.995^500 = 0.082` (8% weight at horizon)

### 2. `gae_lambda` (GAE Lambda)

**What it does**: Trade-off between bias and variance in advantage estimation.
- λ=0: only 1-step TD (low variance, high bias)
- λ=1: full Monte Carlo (high variance, low bias)

**Current**: 0.95

**Recommended**:
- **Navigation**: 0.90 (shorter credit assignment)
- **Battles**: 0.95 (balanced)
- **Badge Quest**: 0.97 (longer credit assignment)

**Reasoning**: Similar to gamma - higher λ helps with long-term credit but adds variance.

### 3. `ent_coef` (Entropy Coefficient)

**What it does**: Encourages exploration by penalizing deterministic policies.

**Current**: 0.01

**Recommended for Sparse Rewards**:
- **Early Training** (0-1M steps): 0.02-0.05 (high exploration)
- **Mid Training** (1-10M steps): 0.01 (balanced)
- **Late Training** (10M+ steps): 0.005 (exploit learned policy)

**Reasoning**:
- With sparse rewards, random exploration is crucial early on
- Too high entropy → agent never exploits
- Too low entropy → agent gets stuck in local optima

**Best Practice**: Anneal entropy over training:
```python
initial_ent = 0.03
final_ent = 0.005
current_ent = final_ent + (initial_ent - final_ent) * (1 - progress)
```

### 4. `learning_rate`

**What it does**: Step size for gradient updates.

**Current**: 3e-4

**Recommended**:
- **Default**: 3e-4 (good starting point)
- **Fine-tuning**: 1e-4 (when resuming from checkpoint)
- **Aggressive**: 5e-4 (if training is too slow)

**Reasoning**: Pokemon Red is complex, but not as hard as Dota/StarCraft. Default LR is fine.

### 5. `n_steps` (Rollout Horizon)

**What it does**: Number of steps collected before each PPO update.

**Current**: 512 (reduced for RAM constraints)

**Recommended**:
- **Ideal**: 2048-4096 (longer rollouts for better advantage estimates)
- **RAM-Constrained**: 512-1024 (current)

**Reasoning**:
- Longer rollouts = better credit assignment for sparse rewards
- But longer rollouts = more RAM (current bottleneck)
- **Compromise**: 1024 steps if RAM allows

### 6. `batch_size`

**What it does**: Minibatch size for gradient updates.

**Current**: 512

**Recommended**:
- Depends on `n_steps × num_envs` (total samples per rollout)
- Rule of thumb: `batch_size` = 1/4 to 1/2 of total samples

**Example**:
```
n_steps = 512
num_envs = 12
total_samples = 512 × 12 = 6144

batch_size options:
- 1024 (6144/6 = 6 minibatches)
- 1536 (6144/4 = 4 minibatches) ← good choice
- 2048 (6144/3 = 3 minibatches)
```

**Recommended**: 1024-1536 for current setup

### 7. `n_epochs`

**What it does**: Number of passes through the rollout buffer.

**Current**: 1

**Recommended**:
- **Fast iteration**: 1-2 epochs (current)
- **Data efficiency**: 4-8 epochs (but risk overfitting)

**Reasoning**: With sparse rewards, more epochs helps extract signal, but risk catastrophic forgetting.

**Best Practice**: Start with 3-4 epochs, reduce if KL divergence spikes.

### 8. `clip_range`

**What it does**: Limits policy updates to prevent destructive changes.

**Current**: 0.2

**Recommended**: 0.1-0.2 (standard range, no need to change)

### 9. `vf_coef` (Value Function Coefficient)

**What it does**: Weight of value loss in total loss.

**Current**: 0.5

**Recommended**: 0.5-1.0 (higher for sparse rewards)

**Reasoning**: With sparse rewards, accurate value function is critical for advantage estimation.

### 10. `max_grad_norm`

**What it does**: Gradient clipping to prevent exploding gradients.

**Current**: 0.5

**Recommended**: 0.5-1.0 (standard, no change needed)

---

## Recommended Hyperparameter Sets

### Set 1: Navigation Task (Dense Rewards)
```json
{
  "n_steps": 512,
  "batch_size": 1024,
  "n_epochs": 3,
  "gamma": 0.99,
  "gae_lambda": 0.90,
  "ent_coef": 0.01,
  "learning_rate": 0.0003,
  "clip_range": 0.2,
  "vf_coef": 0.5
}
```

### Set 2: Battle Training (Sparse Rewards)
```json
{
  "n_steps": 1024,
  "batch_size": 1536,
  "n_epochs": 4,
  "gamma": 0.995,
  "gae_lambda": 0.95,
  "ent_coef": 0.02,
  "learning_rate": 0.0003,
  "clip_range": 0.2,
  "vf_coef": 0.7
}
```

### Set 3: Badge Quest (Very Sparse Rewards)
```json
{
  "n_steps": 2048,
  "batch_size": 2048,
  "n_epochs": 4,
  "gamma": 0.997,
  "gae_lambda": 0.97,
  "ent_coef": 0.015,
  "learning_rate": 0.0003,
  "clip_range": 0.2,
  "vf_coef": 1.0
}
```

---

## Hyperparameter Tuning Strategy

### Step 1: Baseline Run (1-2M steps)

Use default hyperparameters, observe:
- Is KL divergence stable? (<0.05)
- Is clip fraction reasonable? (0.1-0.3)
- Is explained variance increasing? (toward 1.0)
- Is entropy decreasing? (should drop slowly)

### Step 2: Identify Bottleneck

**If learning is too slow**:
- Increase `ent_coef` (more exploration)
- Increase `learning_rate` (faster updates)
- Increase `n_epochs` (more data reuse)

**If policy is unstable**:
- Decrease `learning_rate`
- Decrease `clip_range`
- Decrease `n_epochs`

**If value function is poor** (`explained_variance` < 0.5):
- Increase `vf_coef`
- Increase `n_steps` (better advantage estimates)
- Check reward scaling

### Step 3: Iterate

Make ONE change at a time, run for 5-10M steps, compare to baseline.

---

## Advanced: Entropy Annealing

**Problem**: Fixed entropy is suboptimal - want high exploration early, low exploitation late.

**Solution**: Anneal `ent_coef` over training.

**Implementation** (future work, requires code changes):
```python
def get_entropy_coef(current_step, total_steps, initial_ent=0.03, final_ent=0.005):
    progress = current_step / total_steps
    return final_ent + (initial_ent - final_ent) * (1 - progress)
```

**Usage**:
```python
model.ent_coef = get_entropy_coef(model.num_timesteps, total_timesteps)
```

---

## Monitoring Hyperparameters in TensorBoard

**Key Metrics to Watch**:

### `train/approx_kl`
- **Good**: <0.05 (policy updates are conservative)
- **Warning**: 0.05-0.1 (policy changing quickly)
- **Bad**: >0.1 (policy unstable, reduce LR)

### `train/clip_fraction`
- **Good**: 0.1-0.3 (some updates clipped, not all)
- **Too Low**: <0.1 (clip_range too large or LR too small)
- **Too High**: >0.5 (updates too aggressive, reduce LR)

### `train/entropy_loss`
- **Good**: Slowly decreasing over training
- **Bad**: Drops to zero quickly (policy too deterministic)
- **Bad**: Stays flat (policy not learning)

### `train/explained_variance`
- **Good**: Increasing toward 1.0 (value function learning)
- **Bad**: <0.5 after 10M steps (value function failing)

### `train/value_loss`
- **Good**: Decreasing over training
- **Bad**: Increasing or unstable (value function diverging)

---

## Hardware-Specific Tuning

### For User's Setup (32GB RAM, RTX 4080)

**Constraints**:
- **RAM**: Limited by `agent_stats` accumulation
- **VRAM**: 16GB (plenty for batch processing)
- **CPU**: i9-13900K (24 cores, good for parallel envs)

**Recommendations**:
1. Fix `agent_stats` memory leak (sample every 10 steps, not every step)
2. Once fixed, increase to:
   - `num_envs`: 20-24 (use CPU cores)
   - `n_steps`: 2048 (better credit assignment)
   - `batch_size`: 2048 (utilize VRAM)

**Current Workaround**:
- `num_envs`: 12 (safe for RAM)
- `n_steps`: 512 (minimal RAM footprint)
- `batch_size`: 1024-1536 (good GPU util)

---

## Testing New Hyperparameters

### Quick Test (5M steps, ~1 hour)
```bash
python training/train_ppo.py \
  --config configs/battle_focused_redesign.json \
  --run-name hyperparam_test \
  --num-envs 12 \
  --total-multiplier 100 \
  --batch-size 1536 \
  --n-epochs 4 \
  --ent-coef 0.02 \
  --gamma 0.995
```

**Compare to baseline**: Same config with default hyperparameters.

**Metrics to Compare**:
- `train/episode_return_mean` (higher is better)
- `train/episode_battles_started_mean` (should increase)
- `train/approx_kl` (should be stable <0.05)
- `train/explained_variance` (should increase)

---

## Success Criteria

**Hyperparameters are well-tuned if**:
1. KL divergence remains <0.05 throughout training
2. Clip fraction is 0.1-0.3
3. Explained variance reaches >0.7 by 20M steps
4. Entropy decreases smoothly (not suddenly)
5. Agent shows learning progress (battles increase, returns increase)

**Red Flags**:
- KL spikes >0.1 → reduce learning rate
- Clip fraction >0.5 → reduce learning rate or clip_range
- Explained variance flat at <0.3 → increase n_steps or vf_coef
- Entropy drops to zero in <1M steps → increase ent_coef

---

## Final Recommendations

For the user's stated goal (agent learns to battle), use **Set 2** hyperparameters:

```bash
python training/train_ppo.py \
  --config configs/battle_focused_redesign.json \
  --run-name battle_tuned \
  --num-envs 12 \
  --total-multiplier 200 \
  --preset small
```

With `configs/battle_focused_redesign.json` already containing:
- `n_epochs`: 4
- `gamma`: 0.995
- `ent_coef`: 0.02
- `batch_size`: 1024

**Expected Outcome**: Agent should start encountering battles within 1-2M steps, with increasing win rate by 10M steps.
