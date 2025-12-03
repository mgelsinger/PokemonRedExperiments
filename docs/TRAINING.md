# Pokemon Red PPO Training Guide

This guide explains how to use the improved PPO training system with reward shaping and curriculum learning.

## Table of Contents

- [Overview](#overview)
- [Reward Shaping System](#reward-shaping-system)
- [Curriculum Tasks](#curriculum-tasks)
- [Running Experiments](#running-experiments)
- [Monitoring Training](#monitoring-training)
- [Hyperparameter Tuning](#hyperparameter-tuning)
- [Troubleshooting](#troubleshooting)

---

## Overview

The training system has been significantly enhanced with:

1. **Structured Reward Shaping**: Clear breakdown of rewards into exploration, battle, milestone, and penalty components
2. **Curriculum Learning**: Multiple task presets from simple exploration to full game
3. **Configurable Hyperparameters**: All PPO hyperparameters can be adjusted via config files
4. **Rich TensorBoard Logging**: Detailed metrics for reward components, episode statistics, and more

---

## Reward Shaping System

### Reward Components

The reward system is broken down into four main components:

#### 1. Exploration Rewards
- **New Tile**: Reward for visiting a tile not seen this episode
- **Recent Tile**: Smaller reward for revisiting tiles not seen recently
- **Purpose**: Encourages the agent to explore new areas

#### 2. Battle Rewards
- **HP Delta**: Reward based on `(opponent_hp_loss - player_hp_loss)`
- **Battle Win**: Large positive reward for winning battles
- **Battle Loss**: Negative reward for losing battles
- **Purpose**: Teaches effective combat strategies

#### 3. Milestone Rewards
- **Badge**: Large reward for earning gym badges
- **Level Up**: Reward for gaining Pokemon levels
- **Event Flags**: Reward for triggering game events
- **Key Locations**: Reward for reaching important map locations
- **Purpose**: Provides clear progress signals for long-term objectives

#### 4. Penalty Rewards
- **Step Penalty**: Small negative reward per step (encourages efficiency)
- **Wall Collision**: Penalty for walking into walls
- **Stuck Penalty**: Penalty for staying in the same location too long
- **Purpose**: Discourages inefficient behavior

### Configuring Rewards

Rewards can be configured in three ways:

#### 1. Predefined Configurations

```python
from env.reward_config import get_reward_config

# Available presets:
# - 'default': Balanced rewards
# - 'exploration': Heavy exploration focus
# - 'battle': Heavy battle focus
# - 'milestone': Heavy milestone focus
# - 'minimal_penalty': Reduced penalties

reward_config = get_reward_config('exploration')
```

#### 2. Custom Configuration in JSON

```json
{
  "env": {
    "reward_config": {
      "exploration_new_tile": 5.0,
      "exploration_recent_tile": 0.5,
      "battle_hp_delta": 2.0,
      "battle_win": 50.0,
      "battle_loss": -10.0,
      "milestone_badge": 300.0,
      "milestone_level_up": 5.0,
      "penalty_step": -0.01,
      "penalty_wall": -0.1,
      "reward_scale": 1.0
    }
  }
}
```

#### 3. Programmatic Configuration

```python
from env.reward_config import RewardConfig

config = RewardConfig(
    exploration_new_tile=5.0,
    battle_win=100.0,
    milestone_badge=500.0,
    penalty_step=-0.01
)
```

---

## Curriculum Tasks

The curriculum system provides focused training tasks that build up to the full game.

### Available Tasks

#### 1. Exploration Basic (`exploration_basic.json`)
- **Goal**: Explore Pallet Town and Route 1
- **Focus**: Discovering new tiles
- **Episode Length**: ~81k steps
- **Recommended Training**:
  - Smoke test: 1M steps
  - Early learning: 10M steps
  - Deeper training: 50M steps

#### 2. Battle Training (`battle_training.json`)
- **Goal**: Win battles efficiently
- **Focus**: HP management and combat
- **Episode Length**: ~123k steps
- **Recommended Training**:
  - Smoke test: 2M steps
  - Early learning: 20M steps
  - Deeper training: 100M steps

#### 3. Gym Quest (`gym_quest.json`)
- **Goal**: Reach Pewter City and earn Boulder Badge
- **Focus**: Navigation and first gym challenge
- **Episode Length**: ~246k steps
- **Recommended Training**:
  - Smoke test: 4M steps
  - Early learning: 40M steps
  - Deeper training: 200M steps

#### 4. Full Game Shaped (`full_game_shaped.json`)
- **Goal**: Complete the game
- **Focus**: All aspects with aggressive shaping
- **Episode Length**: ~410k steps
- **Recommended Training**:
  - Smoke test: 10M steps
  - Early learning: 100M steps
  - Deeper training: 500M+ steps

---

## Running Experiments

### Basic Training

```bash
# Train with a specific task config
python training/train_ppo.py \
  --config configs/exploration_basic.json \
  --rom PokemonRed.gb \
  --state init.state \
  --run-name exploration_run_01

# Train with GPU preset (adjusts num_envs and batch_size)
python training/train_ppo.py \
  --config configs/gym_quest.json \
  --preset medium \
  --run-name gym_quest_medium

# Train with custom settings
python training/train_ppo.py \
  --config configs/battle_training.json \
  --num-envs 32 \
  --batch-size 1024 \
  --total-multiplier 2000 \
  --run-name battle_long_run
```

### Resuming Training

```bash
# Resume from latest checkpoint
python training/train_ppo.py \
  --config configs/gym_quest.json \
  --run-name gym_quest_01 \
  --resume-latest

# Resume from specific checkpoint
python training/train_ppo.py \
  --config configs/gym_quest.json \
  --run-name gym_quest_01 \
  --resume-checkpoint runs/gym_quest_01/poke_5000000_steps.zip
```

### Testing Reward Shaping

Before running long training sessions, test that rewards work correctly:

```bash
python tools/test_reward_shaping.py --rom PokemonRed.gb --state init.state
```

This will:
- Test all reward configurations
- Verify exploration rewards work
- Check battle reward setup
- Validate milestone detection
- Confirm penalties are applied

---

## Monitoring Training

### TensorBoard Metrics

Start TensorBoard:
```bash
tensorboard --logdir runs/<run_name>
```

#### Key Metrics to Watch

**Reward Components** (`reward_components/`)
- `exploration`: Should be positive and increase as agent discovers new areas
- `battle`: Should increase as agent learns to win battles
- `milestone`: Should spike when badges/levels are earned
- `penalty`: Will be negative; should decrease in magnitude as agent becomes efficient
- `total_shaped`: Overall episode return from shaped rewards

**Episode Statistics** (`episode/`)
- `length_mean`: Average episode length (should vary based on task)
- `length_max`: Longest episode
- `length_min`: Shortest episode

**PPO Metrics** (default SB3 metrics)
- `train/approx_kl`: KL divergence (should be small, ~0.01-0.05)
- `train/clip_fraction`: Fraction of clipped gradients (0.1-0.3 is good)
- `train/entropy_loss`: Exploration measure (should slowly decrease)
- `train/explained_variance`: How well value function fits (closer to 1 is better)
- `train/learning_rate`: Current learning rate
- `train/policy_gradient_loss`: Policy loss
- `train/value_loss`: Value function loss

**Environment Stats** (`env_stats/`)
- `coord_count`: Number of unique coordinates visited
- `levels_sum`: Total Pokemon levels
- `badge`: Number of badges earned
- `max_map_progress`: Highest map index reached

### Interpreting Results

**Good Learning Signs:**
- Reward components increasing over time
- `explained_variance` approaching 1.0
- `approx_kl` staying small (<0.1)
- Episode length stabilizing
- Coord count increasing (for exploration tasks)

**Bad Learning Signs:**
- All rewards near zero or not changing
- `explained_variance` negative or very low
- `approx_kl` very large (>0.5) or zero
- `clip_fraction` near 0 or 1
- Policy entropy dropping too quickly to 0

---

## Hyperparameter Tuning

### PPO Hyperparameters

All hyperparameters can be set in config JSON files:

```json
{
  "train": {
    "learning_rate": 0.0003,     // Learning rate (try 1e-4 to 5e-4)
    "clip_range": 0.2,            // PPO clip parameter (0.1-0.3)
    "n_epochs": 4,                // Epochs per update (1-10)
    "batch_size": 1024,           // Minibatch size (256-2048)
    "gamma": 0.997,               // Discount factor (0.99-0.999)
    "gae_lambda": 0.95,           // GAE lambda (0.9-0.99)
    "ent_coef": 0.01,             // Entropy coefficient (0.001-0.1)
    "vf_coef": 0.5,               // Value function coefficient (0.5-1.0)
    "max_grad_norm": 0.5,         // Gradient clipping (0.3-1.0)
    "num_envs": 16,               // Parallel environments
    "total_multiplier": 1000      // Training length multiplier
  }
}
```

### Common Adjustments

#### Agent Too Random (High Entropy)
- Decrease `ent_coef` (e.g., 0.01 → 0.005)
- Train longer to let entropy naturally decay

#### Agent Not Learning
- Increase `learning_rate` (e.g., 3e-4 → 5e-4)
- Increase `n_epochs` (e.g., 1 → 4)
- Check reward shaping (are rewards too sparse?)
- Increase `batch_size` and `num_envs`

#### Training Unstable
- Decrease `learning_rate` (e.g., 3e-4 → 1e-4)
- Decrease `clip_range` (e.g., 0.2 → 0.1)
- Lower `max_grad_norm` (e.g., 0.5 → 0.3)

#### Agent Too Conservative
- Increase `ent_coef` (e.g., 0.01 → 0.02)
- Decrease penalties in reward config

---

## Troubleshooting

### Reward Components All Zero

**Issue**: TensorBoard shows all reward components as zero.

**Solutions**:
1. Check that `reward_config` is properly loaded in the environment config
2. Verify the environment is using shaped rewards (not legacy rewards only)
3. Run the test script to verify: `python tools/test_reward_shaping.py`

### Agent Walks in Circles

**Issue**: Agent repeatedly visits the same small area.

**Solutions**:
1. Increase `exploration_new_tile` coefficient
2. Increase `penalty_stuck` (make it more negative)
3. Add higher `penalty_step` to discourage long episodes
4. Reduce `exploration_recent_tile` to discourage backtracking

### Agent Ignores Battles

**Issue**: Agent avoids or performs poorly in battles.

**Solutions**:
1. Increase `battle_win` and `battle_hp_delta` coefficients
2. Use the `battle_training` task config
3. Add higher `milestone_level_up` to encourage leveling up
4. Increase episode length to allow time for battles

### Training Too Slow

**Issue**: Training takes too long or uses too much compute.

**Solutions**:
1. Use a smaller GPU preset: `--preset small`
2. Reduce `max_steps` in environment config
3. Reduce `num_envs` (but keep `batch_size` as multiple of `n_steps * num_envs`)
4. Use `--no-stream` to disable map streaming overhead

### Out of Memory (OOM)

**Issue**: CUDA out of memory error during training.

**Solutions**:
1. Reduce `num_envs` (e.g., 16 → 8)
2. Reduce `batch_size` (e.g., 1024 → 512)
3. Use `--preset small`
4. Close other GPU-using programs

---

## Best Practices

### 1. Start with Curriculum Learning
Begin with simpler tasks before tackling the full game:
1. `exploration_basic` (10M steps)
2. `battle_training` (20M steps)
3. `gym_quest` (40M steps)
4. `full_game_shaped` (100M+ steps)

### 2. Run Smoke Tests First
Test each config with 1-2M steps to verify:
- Rewards are being computed correctly
- Agent shows some learning signal
- No errors or crashes

### 3. Monitor Frequently
Check TensorBoard every few million steps to catch issues early.

### 4. Save Checkpoints Regularly
Default is every `max_steps/2` steps. Can override with `--checkpoint-freq`.

### 5. Use Version Control
Track your config changes and training runs:
```bash
git add configs/
git commit -m "Add new exploration config with higher rewards"
```

### 6. Document Experiments
Keep notes on what works and what doesn't. The UI dashboard at `tools/ui_server.py` helps with this.

---

## Example Workflows

### Quick Test (1M steps, ~10 minutes)
```bash
python training/train_ppo.py \
  --config configs/exploration_basic.json \
  --total-multiplier 100 \
  --run-name quick_test \
  --preset small
```

### Medium Run (20M steps, ~2-4 hours)
```bash
python training/train_ppo.py \
  --config configs/battle_training.json \
  --total-multiplier 500 \
  --run-name battle_medium \
  --preset medium
```

### Long Run (100M steps, ~12-24 hours)
```bash
python training/train_ppo.py \
  --config configs/full_game_shaped.json \
  --total-multiplier 2500 \
  --run-name full_game_long \
  --preset large \
  --eval-every-steps 5000000
```

---

## Further Reading

- [Proximal Policy Optimization Paper](https://arxiv.org/abs/1707.06347)
- [Stable-Baselines3 Documentation](https://stable-baselines3.readthedocs.io/)
- [Reward Shaping Best Practices](https://gibberblot.github.io/rl-notes/single-agent/reward-shaping.html)

---

## Support

For issues or questions:
1. Check the [troubleshooting section](#troubleshooting)
2. Run the test script: `python tools/test_reward_shaping.py`
3. Review TensorBoard metrics
4. Open an issue on the GitHub repository
