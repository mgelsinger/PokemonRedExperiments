# Quick Reference: PPO Training Improvements

## Running Different Tasks

```bash
# Exploration task (easiest)
python training/train_ppo.py --config configs/exploration_basic.json --run-name explore_01

# Battle training
python training/train_ppo.py --config configs/battle_training.json --run-name battle_01

# First gym quest
python training/train_ppo.py --config configs/gym_quest.json --run-name gym_01

# Full game with shaping
python training/train_ppo.py --config configs/full_game_shaped.json --run-name fullgame_01
```

## Testing Reward Shaping

```bash
python tools/test_reward_shaping.py --rom PokemonRed.gb --state init.state
```

## Viewing Results

```bash
tensorboard --logdir runs/<run_name>
```

**Key TensorBoard tabs:**
- `reward_components/` - Exploration, battle, milestone, penalty breakdowns
- `episode/` - Episode length statistics
- `train/` - PPO metrics (KL divergence, entropy, clip fraction)
- `trajectory/` - Exploration maps and visualizations

## Reward Component Configuration

Create custom reward configs in JSON:

```json
{
  "env": {
    "reward_config": {
      "exploration_new_tile": 5.0,      // Reward for new tiles
      "battle_win": 100.0,               // Reward for winning battles
      "milestone_badge": 500.0,          // Reward for earning badges
      "penalty_step": -0.01,             // Small step penalty
      "reward_scale": 1.0                // Global multiplier
    }
  }
}
```

## PPO Hyperparameters

Adjust in config JSON:

```json
{
  "train": {
    "learning_rate": 0.0003,             // Learning rate
    "clip_range": 0.2,                   // PPO clip parameter
    "n_epochs": 4,                       // Update epochs
    "batch_size": 1024,                  // Minibatch size
    "gamma": 0.997,                      // Discount factor
    "ent_coef": 0.01                     // Entropy coefficient
  }
}
```

## Common Issues

**Agent not learning:**
- Increase learning_rate
- Check reward_components/ in TensorBoard (should be non-zero)
- Increase training steps

**Agent too random:**
- Decrease ent_coef
- Train longer (entropy naturally decays)

**Out of memory:**
- Use `--preset small`
- Reduce num_envs or batch_size

**Training too slow:**
- Use `--preset large` (if you have GPU memory)
- Disable streaming with `--no-stream`

## File Structure

```
env/
  reward_config.py         # Reward shaping configuration
  curriculum_tasks.py      # Task definitions
  red_gym_env.py          # Environment with shaped rewards

configs/
  exploration_basic.json  # Exploration task preset
  battle_training.json    # Battle task preset
  gym_quest.json         # Gym quest preset
  full_game_shaped.json  # Full game preset

docs/
  TRAINING.md            # Comprehensive training guide
  QUICK_REFERENCE.md     # This file

tools/
  test_reward_shaping.py # Validation test script
```

## Recommended Training Progression

1. **Test rewards** (5 minutes):
   ```bash
   python tools/test_reward_shaping.py
   ```

2. **Smoke test** (10 minutes, 1M steps):
   ```bash
   python training/train_ppo.py --config configs/exploration_basic.json \
     --total-multiplier 100 --run-name smoke_test
   ```

3. **Early learning** (1-2 hours, 10-20M steps):
   ```bash
   python training/train_ppo.py --config configs/exploration_basic.json \
     --total-multiplier 1000 --run-name exploration_early
   ```

4. **Progress to harder tasks** as agent improves

## See Also

- [docs/TRAINING.md](TRAINING.md) - Complete training guide
- [README.md](../README.md) - Main repository README
