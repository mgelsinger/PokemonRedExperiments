# Debug and Evaluation Guide

This guide explains how to use the debugging and evaluation tools.

---

## Debug Reward Shaping

The `debug_rewards.py` script validates that reward shaping is working correctly by running random actions and printing detailed reward breakdowns.

### Basic Usage

```bash
# Debug a single episode with default task
python debug_rewards.py

# Debug a specific task
python debug_rewards.py --task walk_to_pokecenter

# Debug for a specific number of steps
python debug_rewards.py --task exploration_basic --steps 500

# Debug multiple episodes
python debug_rewards.py --task battle_training --episodes 5
```

### What It Shows

For each step or episode, the script prints:

1. **Total Reward** - Sum of all components
2. **Reward Components**:
   - `Exploration` - Reward for visiting new tiles
   - `Battle` - Reward from HP delta and wins/losses
   - `Milestone` - Reward for badges, levels, events
   - `Penalty` - Negative rewards (step cost, wall collisions, stuck)
   - `Legacy` - Old reward system (for comparison)
3. **Game State**:
   - Position (x, y, map_id)
   - HP fraction
   - Badge count
   - Tiles explored this episode
4. **Battle Stats**:
   - In battle status
   - Battles won/lost this episode

### Example Output

```
================================================================================
DEBUGGING REWARDS: walk_to_pokecenter
Running 1 random episodes
================================================================================

Task Config:
  Max Steps:    40960
  Termination:  pokecenter_reached

Reward Config:
  Exploration:  new=10.0, recent=1.0
  Battle:       hp_delta=0.0, win=0.0, loss=0.0
  Milestone:    badge=0.0, level=0.0, location=50.0
  Penalty:      step=-0.001, wall=-0.1, stuck=-0.2

================================================================================
EPISODE 1/1
================================================================================
Initial Position: (6, 5, 0)

  Step 50: reward=42.50, tiles=23

Episode 1 Summary:
  Length:          124
  Total Reward:    89.2450
  Reward Components:
    exploration : 85.0000
    battle      :  0.0000
    milestone   :  0.0000
    penalty     : -0.1240
  Tiles Explored:  85
  Battles Won:     0
  Battles Lost:    0

================================================================================
OVERALL SUMMARY (1 episodes)
================================================================================
Mean Return:         89.2450 ± 0.0000
Mean Length:         124.0 ± 0.0000
Max Return:          89.2450
Min Return:          89.2450
```

### Command-Line Options

```
--task <name>       Task name (config file without .json)
--rom <path>        Path to Pokemon Red ROM (default: PokemonRed.gb)
--state <path>      Path to initial state (default: init.state)
--steps <n>         Run N steps in a single episode
--episodes <n>      Run N complete episodes
```

**Note**: `--steps` and `--episodes` are mutually exclusive. If neither is provided, runs 1 episode.

---

## Evaluate Trained Policy

The `eval_policy.py` script evaluates a trained PPO checkpoint by running deterministic episodes and measuring performance.

### Basic Usage

```bash
# Evaluate with default settings (10 episodes)
python eval_policy.py \
  --config configs/walk_to_pokecenter.json \
  --checkpoint runs/my_run/poke_500000_steps.zip

# Evaluate more episodes
python eval_policy.py \
  --config configs/gym_quest.json \
  --checkpoint runs/gym_run/poke_5000000_steps.zip \
  --n_episodes 20

# Export detailed trajectories
python eval_policy.py \
  --config configs/walk_to_pokecenter.json \
  --checkpoint runs/my_run/final.zip \
  --n_episodes 10 \
  --export_trajectory
```

### What It Shows

For each episode:
1. **Start/End Position** - (x, y, map_id)
2. **Episode Length** - Number of steps taken
3. **Episode Return** - Total reward accumulated
4. **Success** - Whether task objective was achieved
5. **Reward Breakdown** - Components (exploration, battle, milestone, penalty)

Overall statistics:
1. **Mean Return** ± std deviation
2. **Mean Length** ± std deviation
3. **Success Rate** - Percentage of successful episodes
4. **Max/Min Return** - Best and worst episodes

### Example Output

```
Running 10 evaluation episodes...
Max steps per episode: 40960

Episode 1/10:
  Start position: (6, 5, 0)
    Step 100: reward=95.20, tiles=92
  End position:   (19, 15, 40)
  Episode length: 156
  Episode return: 112.4560
  Success:        ✓
  Reward breakdown:
    exploration : 95.2000
    battle      :  0.0000
    milestone   : 50.0000
    penalty     : -0.1560

...

================================================================================
EVALUATION RESULTS
================================================================================
Episodes:            10
Mean Return:         98.3421 ± 12.5432
Mean Length:         203.4 ± 45.2
Success Rate:        80.00%
Max Return:          112.4560
Min Return:          67.8900

Per-Episode Results:
  Episode  1: return=112.4560, length= 156, success=✓
  Episode  2: return= 89.2340, length= 245, success=✓
  Episode  3: return= 67.8900, length= 312, success=✗
  ...

Evaluation completed in 45.2 seconds

Results saved to: eval_results_20250102_143522.json
```

### Command-Line Options

```
--config <path>         Path to task config JSON (required)
--checkpoint <path>     Path to trained model .zip (required)
--rom <path>            Path to Pokemon Red ROM (default: PokemonRed.gb)
--state <path>          Path to initial state (default: init.state)
--n_episodes <n>        Number of episodes to run (default: 10)
--max_steps <n>         Max steps per episode (default: from config)
--export_trajectory     Export detailed step-by-step trajectories
--output <path>         Output JSON file (default: eval_results_<timestamp>.json)
--seed <n>              Random seed (default: 42)
```

### Output JSON Format

The evaluation results are saved to a JSON file with this structure:

```json
{
  "n_episodes": 10,
  "mean_return": 98.3421,
  "std_return": 12.5432,
  "mean_length": 203.4,
  "std_length": 45.2,
  "success_rate": 0.8,
  "max_return": 112.456,
  "min_return": 67.89,
  "episode_returns": [112.456, 89.234, ...],
  "episode_lengths": [156, 245, ...],
  "successes": [true, true, false, ...],
  "config": "walk_to_pokecenter",
  "checkpoint": "runs/my_run/poke_500000_steps.zip",
  "timestamp": "2025-01-02 14:35:22",
  "elapsed_time": 45.2
}
```

If `--export_trajectory` is used, each episode also includes a `trajectories` array with step-by-step details.

---

## Task-Specific Success Conditions

### walk_to_pokecenter
- **Success**: Reaching Viridian Pokecenter (map ID 40)
- **Expected Length**: 100-300 steps
- **Expected Return**: 50-150 (depends on exploration)

### gym_quest
- **Success**: Earning Boulder Badge (badge count increases)
- **Expected Length**: 5000-20000 steps
- **Expected Return**: 500-2000+

### exploration_basic
- **Success**: Not defined (runs to max_steps)
- **Expected Length**: Full episode length
- **Expected Return**: Depends on tiles explored

### battle_training
- **Success**: Not defined (runs to max_steps)
- **Expected Length**: Full episode length
- **Expected Return**: Depends on battles won

---

## Workflow: From Training to Evaluation

### 1. Train a Policy

```bash
python training/train_ppo.py \
  --config configs/walk_to_pokecenter.json \
  --run-name pokecenter_01 \
  --total-multiplier 500 \
  --preset medium
```

### 2. Monitor Training

```bash
tensorboard --logdir runs/pokecenter_01
```

Check:
- `reward_components/exploration` is increasing
- `episode/length_mean` is decreasing (agent finding goal faster)
- `train/approx_kl` is small (<0.1)

### 3. Debug Rewards (Optional)

Before evaluating, verify rewards make sense:

```bash
python debug_rewards.py --task walk_to_pokecenter --episodes 3
```

### 4. Evaluate Checkpoint

```bash
python eval_policy.py \
  --config configs/walk_to_pokecenter.json \
  --checkpoint runs/pokecenter_01/poke_2000000_steps.zip \
  --n_episodes 20
```

### 5. Compare Checkpoints

Evaluate multiple checkpoints to find the best:

```bash
# Early checkpoint
python eval_policy.py --config configs/walk_to_pokecenter.json \
  --checkpoint runs/pokecenter_01/poke_500000_steps.zip \
  --n_episodes 20 --output eval_500k.json

# Mid checkpoint
python eval_policy.py --config configs/walk_to_pokecenter.json \
  --checkpoint runs/pokecenter_01/poke_2000000_steps.zip \
  --n_episodes 20 --output eval_2m.json

# Final checkpoint
python eval_policy.py --config configs/walk_to_pokecenter.json \
  --checkpoint runs/pokecenter_01/final.zip \
  --n_episodes 20 --output eval_final.json
```

Then compare the results in the JSON files.

---

## Troubleshooting

### debug_rewards.py

**Problem**: Rewards are all zero

**Solution**:
- Check that `reward_config` is properly set in the task JSON
- Verify task JSON file exists in `configs/` directory
- Run with `--task` matching an existing config file

**Problem**: Script crashes with import error

**Solution**:
- Ensure you're in the project root directory
- Check that all dependencies are installed: `pip install -r requirements.txt`
- Verify ROM and state files exist

### eval_policy.py

**Problem**: "Error loading model"

**Solution**:
- Verify checkpoint file exists and is a valid `.zip` file
- Check that checkpoint was saved from the same environment
- Try loading a different checkpoint

**Problem**: Success rate is 0%

**Solution**:
- Check if the policy was trained long enough
- Verify termination condition matches the task
- Try evaluating more episodes (`--n_episodes 50`)
- Check if reward shaping is configured correctly

**Problem**: Episodes are very short/long

**Solution**:
- Check `--max_steps` parameter
- Verify task config has correct `max_steps`
- Check if termination condition is triggering correctly

---

## See Also

- [TRAINING.md](TRAINING.md) - Full training guide
- [SMOKE_TEST_GUIDE.md](../SMOKE_TEST_GUIDE.md) - Quick testing guide
- [VERIFICATION_AUDIT.md](../VERIFICATION_AUDIT.md) - Implementation status
