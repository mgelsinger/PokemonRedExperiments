# PPO Training Improvements - Change Summary

This document summarizes all improvements made to the Pokemon Red PPO training system.

---

## Overview

The training system has been completely overhauled to provide structured reward shaping, curriculum learning, and comprehensive monitoring. The agent should now learn useful behavior instead of walking randomly.

---

## Major Changes

### 1. Reward Shaping System

**New Files:**
- `env/reward_config.py` - Reward configuration dataclass and presets

**Key Features:**
- **Structured reward breakdown** into four components:
  - **Exploration**: Rewards for visiting new tiles
  - **Battle**: Rewards for HP management and wins/losses
  - **Milestone**: Rewards for badges, levels, events
  - **Penalty**: Penalties for inefficiency (step cost, wall collisions, getting stuck)

- **Configurable coefficients** for each reward component
- **Five preset configurations**: default, exploration, battle, milestone, minimal_penalty
- **Easy customization** via JSON or Python

**Modified Files:**
- `env/red_gym_env.py`:
  - Added `RewardConfig` integration
  - Added episode-specific tracking (tiles visited this episode)
  - Added battle state tracking (HP deltas, win/loss detection)
  - Added milestone tracking (level ups, badges, events)
  - Added penalty detection (wall collisions, stuck detection)
  - Refactored `update_reward()` to use component-based system
  - Added methods: `compute_exploration_reward()`, `compute_battle_reward()`, `compute_milestone_reward()`, `compute_penalty_reward()`
  - Added helper methods: `get_party_levels()`, `get_opponent_hp_fraction()`

### 2. Enhanced TensorBoard Logging

**Modified Files:**
- `training/tensorboard_callback.py`:
  - Added logging for all reward components
  - Added episode length statistics (mean, max, min)
  - Added reward distributions and histograms
  - Added total shaped reward tracking

**New TensorBoard Metrics:**
- `reward_components/exploration` - Exploration reward per episode
- `reward_components/battle` - Battle reward per episode
- `reward_components/milestone` - Milestone reward per episode
- `reward_components/penalty` - Penalty per episode
- `reward_components/total_shaped` - Total shaped reward (sum of all components)
- `episode/length_mean` - Average episode length
- `episode/length_max` - Maximum episode length
- `episode/length_min` - Minimum episode length
- `reward_distribs/*` - Histograms of reward distributions

### 3. Curriculum Task System

**New Files:**
- `env/curriculum_tasks.py` - Task definitions with configs

**Available Tasks:**
1. **exploration_basic** - Focus on exploring Pallet Town and Route 1
2. **battle_training** - Focus on winning battles efficiently
3. **gym_quest** - Navigate to first gym and earn Boulder Badge
4. **early_game** - Complete early game objectives
5. **full_game** - Full game playthrough
6. **full_game_shaped** - Full game with aggressive reward shaping

Each task includes:
- Custom reward configuration
- Appropriate episode length
- Recommended training steps

### 4. Configurable PPO Hyperparameters

**Modified Files:**
- `configs/train_default.json` - Added all PPO hyperparameters
- `training/train_ppo.py` - Extract and use all hyperparameters

**Now Configurable:**
- `learning_rate` - Learning rate (default: 3e-4)
- `clip_range` - PPO clip parameter (default: 0.2)
- `vf_coef` - Value function coefficient (default: 0.5)
- `max_grad_norm` - Gradient clipping (default: 0.5)
- `gae_lambda` - GAE lambda (default: 0.95)
- `use_sde` - State-dependent exploration (default: false)
- `normalize_advantage` - Normalize advantages (default: true)
- `target_kl` - Target KL divergence (default: null)

All can be set in config JSON files without modifying code.

### 5. Experiment Preset Configs

**New Files:**
- `configs/exploration_basic.json`
- `configs/battle_training.json`
- `configs/gym_quest.json`
- `configs/full_game_shaped.json`

Each preset includes:
- Task-specific reward configuration
- Optimized PPO hyperparameters
- Recommended training steps (smoke test, early learning, deeper training)

### 6. Validation and Testing

**New Files:**
- `tools/test_reward_shaping.py` - Comprehensive test script

**Tests:**
- Loads all reward configurations
- Tests exploration rewards (verifies new tile detection)
- Tests battle reward setup
- Tests milestone detection
- Tests penalty application
- Prints detailed reward breakdowns

Run with:
```bash
python tools/test_reward_shaping.py --rom PokemonRed.gb --state init.state
```

### 7. Documentation

**New Files:**
- `docs/TRAINING.md` - Comprehensive training guide
- `docs/QUICK_REFERENCE.md` - Quick reference for common tasks
- `docs/CHANGES.md` - This file

**Updated Files:**
- `README.md` - Added overview of new features and curriculum tasks

**Documentation Includes:**
- Detailed explanation of reward shaping system
- How to use curriculum tasks
- TensorBoard monitoring guide
- Hyperparameter tuning advice
- Troubleshooting common issues
- Example workflows
- Best practices

---

## Key Improvements for Learning

### Why the Agent Should Learn Better Now

1. **Dense Rewards**: Instead of sparse legacy rewards, the agent gets immediate feedback for:
   - Visiting new areas (exploration)
   - Making progress in battles (HP delta)
   - Hitting milestones (badges, levels)

2. **Clear Signal**: Reward components are separated, making it easier to debug and tune:
   - If agent doesn't explore: increase exploration coefficients
   - If agent avoids battles: increase battle rewards
   - If agent is inefficient: increase penalties

3. **Curriculum**: Start with easier tasks to build basic skills:
   - Exploration teaches movement
   - Battle training teaches combat
   - Gym quest combines both
   - Full game applies everything

4. **Proper Hyperparameters**: All PPO settings are now exposed and can be tuned for this specific environment.

5. **Visibility**: Rich TensorBoard logging makes it obvious when training is working or not.

---

## Migration Guide

### For Existing Training Runs

Old training runs using legacy rewards will still work. The environment computes both:
- **Shaped rewards** (new system) - returned to agent
- **Legacy rewards** (old system) - logged for comparison

To use new reward shaping on an existing run:
1. Add `"reward_config"` to your environment config
2. Resume training with `--resume-latest`

### Converting Old Configs

Old config:
```json
{
  "env": {
    "reward_scale": 0.5,
    "explore_weight": 0.25
  }
}
```

New config (equivalent):
```json
{
  "env": {
    "reward_config": {
      "exploration_new_tile": 1.0,
      "reward_scale": 0.5
    }
  }
}
```

Or use a preset:
```json
{
  "env": {
    "reward_config": "exploration"
  }
}
```

---

## File Structure

```
PokemonRedExperiments/
├── env/
│   ├── red_gym_env.py           # Modified: Added reward shaping
│   ├── reward_config.py          # NEW: Reward configuration
│   ├── curriculum_tasks.py       # NEW: Task definitions
│   └── ...
├── training/
│   ├── train_ppo.py             # Modified: Added hyperparameters
│   ├── tensorboard_callback.py   # Modified: Enhanced logging
│   └── ...
├── configs/
│   ├── train_default.json        # Modified: Added hyperparameters
│   ├── exploration_basic.json    # NEW: Exploration task preset
│   ├── battle_training.json      # NEW: Battle task preset
│   ├── gym_quest.json           # NEW: Gym quest preset
│   └── full_game_shaped.json    # NEW: Full game preset
├── tools/
│   ├── test_reward_shaping.py   # NEW: Validation tests
│   └── ...
├── docs/
│   ├── TRAINING.md              # NEW: Comprehensive guide
│   ├── QUICK_REFERENCE.md       # NEW: Quick reference
│   └── CHANGES.md               # NEW: This file
└── README.md                    # Modified: Added overview
```

---

## Next Steps

### Immediate Actions

1. **Test the system**:
   ```bash
   python tools/test_reward_shaping.py --rom PokemonRed.gb --state init.state
   ```

2. **Run a smoke test** (10 minutes):
   ```bash
   python training/train_ppo.py --config configs/exploration_basic.json \
     --total-multiplier 100 --run-name smoke_test --preset small
   ```

3. **Monitor in TensorBoard**:
   ```bash
   tensorboard --logdir runs/smoke_test
   ```

4. **Check that rewards are working**:
   - `reward_components/exploration` should be positive and increasing
   - `episode/length_mean` should stabilize
   - `train/entropy_loss` should slowly decrease

### Recommended Training Progression

1. **Exploration Basic** (10-20M steps):
   - Teaches basic movement and map navigation
   - Should see increasing coord_count and exploration rewards

2. **Battle Training** (20-40M steps):
   - Teaches combat mechanics
   - Should see increasing battle wins and level ups

3. **Gym Quest** (40-100M steps):
   - Combines exploration and combat
   - Goal: earn first badge

4. **Full Game Shaped** (100M+ steps):
   - Long-term objective
   - May require hundreds of millions of steps

### Tuning Tips

- **Agent explores too much**: Decrease `exploration_new_tile`, increase `battle` rewards
- **Agent avoids battles**: Increase `battle_win` and `battle_hp_delta`
- **Agent gets stuck**: Increase `penalty_stuck` (more negative)
- **Training unstable**: Decrease `learning_rate`, decrease `clip_range`
- **Agent too random**: Decrease `ent_coef`

---

## Technical Details

### Reward Component Tracking

The environment now tracks episode-specific state:
- `episode_visited_tiles`: Set of tiles visited this episode
- `recent_tile_queue`: Rolling window of recently visited tiles
- `prev_position`: Previous position for wall detection
- `in_battle`, `prev_player_hp`, `prev_opponent_hp`: Battle state
- `prev_levels`, `prev_badges`, `prev_events`: Milestone tracking
- `episode_reward_components`: Accumulated rewards per component

### Reward Computation Flow

1. `step(action)` is called
2. `update_reward()` calls each component method:
   - `compute_exploration_reward()` - checks for new tiles
   - `compute_battle_reward()` - tracks HP and battles
   - `compute_milestone_reward()` - detects achievements
   - `compute_penalty_reward()` - applies penalties
3. Components are summed and returned to agent
4. Legacy rewards are computed in parallel for logging
5. All components are accumulated in `episode_reward_components`

### TensorBoard Logging Flow

1. When episode ends (done=True), callback is triggered
2. `episode_reward_components` is retrieved from all environments
3. Mean and distributions are computed
4. Logged to TensorBoard under:
   - `reward_components/` for means
   - `reward_distribs/` for histograms

---

## Known Limitations

1. **Battle rewards depend on accurate HP tracking**:
   - Opponent HP reading may be unreliable in some battle states
   - Win/loss detection based on HP transitions

2. **Milestone detection is basic**:
   - Level ups detected by sum of party levels
   - Some events may not trigger correctly

3. **Wall detection is simple**:
   - Checks if position didn't change
   - May have false positives in battles/menus

4. **No termination conditions yet**:
   - All tasks run to max_steps
   - Task-specific termination (e.g., "end when badge earned") not implemented

5. **Legacy rewards still computed**:
   - For backward compatibility
   - Slight performance overhead

---

## Future Improvements

Potential enhancements:
- Task-specific termination conditions
- Opponent HP tracking improvements
- More sophisticated milestone detection
- Dynamic reward scaling based on progress
- Entropy scheduling (automatic decay)
- Curriculum auto-advancement
- Multi-objective optimization
- Intrinsic motivation (curiosity, empowerment)

---

## Questions?

See the full training guide at [docs/TRAINING.md](TRAINING.md) or check the troubleshooting section.
