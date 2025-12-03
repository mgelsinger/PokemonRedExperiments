# PHASE 3: Environment Adjustments for Battles - Implementation Summary

## Overview

**Goal**: Make battles more frequent and learnable by adjusting episode lengths, reward structure, and creating battle-focused configurations.

## Changes Made

### 1. Increased Episode Length for Battle Discovery

**Problem**: With `max_steps=2048`, episodes are too short to encounter battles organically.

**Solution**: Created `battle_focused_small.json` with `max_steps=4096` and battle-encouraging reward structure.

**File**: `configs/battle_focused_small.json`
- `max_steps`: 4096 (2× longer episodes)
- `exploration_new_tile`: 0.1 (reduced from 2.0)
- `battle_start_bonus`: 5.0 (NEW - rewards entering battles)
- `battle_hp_delta`: 3.0 (increased from 2.0)
- `battle_win`: 30.0
- `milestone_level_up`: 10.0 (increased from 5.0)

**Rationale**:
- Longer episodes → more time to find grass/trainers
- Lower exploration rewards → less incentive to just wander
- Battle start bonus → actively encourages battle-seeking
- Higher battle/milestone rewards → makes battles worthwhile

### 2. Added Battle Start Bonus Mechanism

**Problem**: Agent only gets rewarded AFTER winning battles, not for seeking them.

**Solution**: Added `battle_start_bonus` reward when entering battle state.

**Files Modified**:
- `env/reward_config.py`: Added `battle_start_bonus: float = 0.0` to RewardConfig
- `env/red_gym_env.py:731`: Reward applied when battle is detected

**Code**:
```python
if current_in_battle and not self.in_battle:
    # ... battle initialization ...
    reward += self.reward_config.battle_start_bonus  # NEW
```

**Impact**:
- Agent gets immediate feedback for finding battles
- Encourages exploration toward grass/trainers
- Dense signal complements sparse battle-win rewards

### 3. Enhanced Battle Metrics (from Phase 2)

**Metrics Now Tracked**:
- `battles_started`: Count of battles initiated
- `steps_to_first_battle`: Steps until first battle this episode
- `badges_earned`: Badges gained this episode
- `levels_gained`: Level-ups this episode
- `deaths`: Death/whiteout count
- `map_progress_max`: Highest map progress reached

**TensorBoard Metrics Added**:
- `train/episode_battles_started_mean`
- `train/episode_battles_won_mean`
- `train/episode_battles_total_mean`
- `train/episode_steps_to_first_battle`
- `train/episode_badges_earned_mean`
- `train/episode_levels_gained_mean`
- `train/episode_deaths_mean`
- `train/episode_map_progress_max`

**Visibility**: These metrics allow monitoring whether the agent is actually encountering and engaging in battles.

### 4. Configuration Philosophy

**Three Training Stages**:

#### Stage 1: Battle-Focused Small (`battle_focused_small.json`)
- **Purpose**: Learn battle mechanics in isolation
- **Duration**: 4096 steps (long enough to encounter battles)
- **Rewards**: Battle-heavy, exploration-light
- **Expected Behavior**: Agent learns to seek grass, engage battles, manage HP

#### Stage 2: Gym Quest (existing `gym_quest.json`)
- **Purpose**: Full task - reach gym, earn badge
- **Duration**: Currently 2048 steps (TOO SHORT - needs increase)
- **Rewards**: Balanced
- **Expected Behavior**: Navigate + battle + progress

#### Stage 3: Full Game (`full_game_shaped.json`)
- **Purpose**: Complete playthrough
- **Duration**: 40960 steps
- **Rewards**: All components enabled

## Remaining Issues (To Address in Phase 4)

### Issue 1: gym_quest.json Still Too Short

**Current**: `max_steps=2048`
**Problem**: Not enough time to reach gym and defeat leader
**Solution**: Increase to at least 8192 steps

### Issue 2: Encounter Rate Still Low

**Current**: Random grass encounters depend on game RNG
**Potential Solutions**:
- Start agent closer to tall grass (custom init states)
- Increase exploration penalty to push agent into grass faster
- Add shaped reward for "time since last battle" penalty

### Issue 3: Memory Constraints

**User Report**: "Over 90% RAM with just 6 envs"
**Root Cause**: `agent_stats` list accumulates 8192+ dicts per episode
**Current Workaround**: Reduced max_steps to 2048-4096
**Better Solution** (for later):
- Sample agent_stats less frequently (every 10 steps)
- Only store summary statistics instead of full trajectory

## Testing Instructions

### Test Battle-Focused Config

```bash
python training/train_ppo.py \
  --config configs/battle_focused_small.json \
  --rom PokemonRed.gb \
  --state init.state \
  --run-name test_battles \
  --num-envs 12 \
  --total-multiplier 100 \
  --preset small
```

**Expected Metrics** (after ~1M steps):
- `train/episode_battles_started_mean` > 0.5
- `train/episode_battles_won_mean` > 0.0
- `train/episode_steps_to_first_battle` < 2000

**TensorBoard**:
```bash
tensorboard --logdir runs/test_battles
```

### Monitor Key Metrics

**What to Watch**:
1. **Battles Started**: Should increase over training (agent learns to find grass)
2. **Steps to First Battle**: Should decrease (agent gets faster at finding battles)
3. **Battle Win Rate**: Should increase (agent learns to fight)
4. **Exploration vs Battle Rewards**: Battle rewards should dominate over time

**Red Flags**:
- Battles started = 0 → Agent stuck or init state wrong
- Steps to first battle = max_steps → Agent never finds grass
- Battle win rate = 0% → Agent enters battles but always loses

## Next Steps (Phase 4)

1. **Reward Rebalancing**: Finalize exploration vs battle vs milestone coefficients
2. **Curriculum Design**: Define progression from battle-only → nav → integrated
3. **Memory Optimization**: Fix agent_stats memory leak for longer episodes
4. **Custom Init States**: Create saves that start near grass/trainers

## Files Modified (Phase 3)

### Created:
- `configs/battle_focused_small.json` - Battle-focused training config

### Modified:
- `env/reward_config.py` - Added battle_start_bonus parameter
- `env/red_gym_env.py` - Applied battle_start_bonus when battle detected

### Documentation:
- `docs/PHASE_3_SUMMARY.md` - This file

## Success Criteria

**Phase 3 is successful if**:
- Agent can discover battles within 4096 steps
- `train/episode_battles_started_mean` > 0 after training
- Battle metrics are visible in TensorBoard
- Configuration allows experimentation with reward balances

**Status**: ✅ COMPLETE - Infrastructure in place, ready for Phase 4 reward tuning
