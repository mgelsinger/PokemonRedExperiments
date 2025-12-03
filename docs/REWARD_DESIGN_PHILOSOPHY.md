# Reward Design Philosophy

## Problem Statement

**Current Issue**: Agent learns to wander instead of battle/progress.

**Root Cause**: Reward imbalance - exploration is dense and frequent, battles are sparse and rare.

## Reward Math Analysis

### Scenario 1: Pure Wandering (Current Behavior)

```
Episode Length: 2048 steps
Action: Walk around, discover ~2000 new tiles

Exploration Reward: 2000 tiles × 2.0 = +4000
Step Penalty: 2048 × -0.002 = -4.096
Total: ~+3995
```

**Agent learns**: "Walk around = big reward"

### Scenario 2: Battle-Seeking (Desired Behavior)

```
Episode Length: 4096 steps
Action: Walk to grass (500 steps), battle (100 steps), repeat 3×

Phase 1: Find grass (500 steps)
  Exploration: 50 new tiles × 2.0 = +100
  Step penalty: 500 × -0.002 = -1.0
  Subtotal: +99

Phase 2: Battle #1 (100 steps)
  Battle start: +5.0
  HP delta: average +15.0 (opponent takes more damage)
  Battle win: +30.0
  Step penalty: 100 × -0.002 = -0.2
  Subtotal: +49.8

Repeat 3×: 3 × (99 + 49.8) = +446.4

Current Total: +446 << +3995 (wandering wins!)
```

**Problem**: Battle path gives 9× less reward than wandering.

---

## Redesigned Reward Structure

### Goal: Make 1 battle worth more than exploring 100 tiles

**New Coefficients**:

```python
exploration_new_tile = 0.1      # Was 2.0 (20× reduction)
exploration_recent_tile = 0.01  # Was 0.2 (20× reduction)

battle_start_bonus = 10.0       # NEW (immediate feedback)
battle_hp_delta = 5.0           # Was 2.0 (2.5× increase)
battle_win = 50.0               # Was 30.0 (1.7× increase)
battle_loss = -10.0             # Was -5.0 (2× penalty)

milestone_level_up = 20.0       # Was 5.0 (4× increase)
milestone_badge = 500.0         # Unchanged (already high)

penalty_step = -0.001           # Was -0.002 (2× less harsh)
```

### Recalculated Scenarios

#### Scenario A: Pure Wandering (Discouraged)
```
2000 tiles × 0.1 = +200
2048 steps × -0.001 = -2.048
Total: ~+198
```

#### Scenario B: Battle-Seeking (Encouraged)
```
Find grass: 50 tiles × 0.1 = +5
Battle start: +10.0
HP delta: +25 (5.0 × 5 HP advantage)
Battle win: +50.0
Total per battle: ~+90

3 battles in 4096 steps: 3 × 90 = +270
Extra exploration: 100 tiles × 0.1 = +10
Step penalty: 4096 × -0.001 = -4.1
Episode Total: ~+276
```

**Result**: Battles now give 40% more reward than pure wandering! ✅

---

## Reward Component Breakdown

### 1. Exploration Rewards

**Philosophy**: Exploration should guide toward interesting areas, not be the primary objective.

**Coefficients**:
- `exploration_new_tile`: **0.1** - Small reward for discovery
- `exploration_recent_tile`: **0.01** - Tiny reward for revisiting
- `exploration_recent_window`: 50 steps

**Expected Behavior**:
- Agent explores to find grass/trainers/items
- Once found, switches to battle/interaction mode
- Doesn't get stuck in "wandering loops"

### 2. Battle Rewards

**Philosophy**: Battles are the core gameplay - should be highly rewarded at all stages (seek, engage, win).

**Coefficients**:
- `battle_start_bonus`: **10.0** - Immediate reward for finding battles
- `battle_hp_delta`: **5.0** - Reward efficient damage dealing
- `battle_win`: **50.0** - Big bonus for winning
- `battle_loss`: **-10.0** - Moderate penalty (not too harsh, or agent avoids battles)

**Expected Behavior**:
- Agent actively seeks tall grass
- Learns to maximize damage while minimizing HP loss
- Balances risk (battle) vs safety (flee)

### 3. Milestone Rewards

**Philosophy**: Key achievements should be celebrated with large rewards to provide long-term goals.

**Coefficients**:
- `milestone_badge`: **500.0** - Huge reward (10× a battle win)
- `milestone_level_up`: **20.0** - Significant reward (encourages grinding)
- `milestone_key_location`: **30.0** - Reward for reaching gyms/centers
- `milestone_event`: **5.0** - Small reward for story progress

**Expected Behavior**:
- Agent learns long-term planning (reach gym, level up for gym leader)
- Balances grinding (level up) vs progress (advance story)

### 4. Penalty Rewards

**Philosophy**: Discourage inefficiency without being too punitive.

**Coefficients**:
- `penalty_step`: **-0.001** - Very light penalty (adds up over 4000 steps)
- `penalty_wall`: **-0.05** - Moderate penalty for walking into walls
- `penalty_stuck`: **-0.1** - Penalty for staying in same tile >600 times

**Expected Behavior**:
- Agent prefers shorter paths
- Avoids getting stuck in corners
- Learns efficient navigation

---

## Configuration Presets

### Preset 1: Battle Training (`battle_focused_redesign.json`)

**Purpose**: Teach agent to seek and win battles

**Coefficients**:
```json
{
  "exploration_new_tile": 0.05,
  "battle_start_bonus": 15.0,
  "battle_hp_delta": 7.0,
  "battle_win": 60.0,
  "battle_loss": -5.0,
  "milestone_level_up": 30.0,
  "penalty_step": -0.0005
}
```

**Reward Ratio**: Battles:Exploration = 100:1

### Preset 2: Exploration + Battles (`balanced_redesign.json`)

**Purpose**: Balanced training for full game

**Coefficients**:
```json
{
  "exploration_new_tile": 0.1,
  "battle_start_bonus": 10.0,
  "battle_hp_delta": 5.0,
  "battle_win": 50.0,
  "battle_loss": -10.0,
  "milestone_level_up": 20.0,
  "milestone_badge": 500.0,
  "penalty_step": -0.001
}
```

**Reward Ratio**: Battles:Exploration = 20:1

### Preset 3: Milestone Rush (`milestone_focused_redesign.json`)

**Purpose**: Speed-run to first badge

**Coefficients**:
```json
{
  "exploration_new_tile": 0.02,
  "battle_start_bonus": 5.0,
  "battle_win": 40.0,
  "milestone_badge": 1000.0,
  "milestone_level_up": 50.0,
  "milestone_key_location": 100.0,
  "penalty_step": -0.002
}
```

**Reward Ratio**: Milestones:Battles:Exploration = 100:10:1

---

## Implementation Strategy

### Step 1: Create Redesigned Configs

- `configs/battle_focused_redesign.json` - Battle training
- `configs/balanced_redesign.json` - Full game balanced
- `configs/milestone_focused_redesign.json` - Badge rush

### Step 2: Test with Short Runs

```bash
python training/train_ppo.py \
  --config configs/battle_focused_redesign.json \
  --run-name test_rewards \
  --num-envs 12 \
  --total-multiplier 50 \
  --preset small
```

**Watch For**:
- `reward_components/exploration` should be LOW
- `reward_components/battle` should be HIGH
- `train/episode_battles_started_mean` should increase

### Step 3: Iterate Based on Metrics

**If battles still rare**:
- Increase `battle_start_bonus` to 20.0
- Reduce `exploration_new_tile` to 0.02

**If agent avoids battles after losses**:
- Reduce `battle_loss` penalty to -3.0
- Increase `battle_start_bonus` (make seeking worth the risk)

**If agent doesn't progress**:
- Increase `milestone_key_location` to 50.0
- Add progress-based shaped rewards

---

## Expected Training Curves

### Phase 1: Random Exploration (0-100k steps)
- `exploration_r` dominates
- `battle_r` near zero
- Agent wanders randomly

### Phase 2: Battle Discovery (100k-500k steps)
- `battles_started` increases
- `steps_to_first_battle` decreases
- Agent learns grass → battles

### Phase 3: Battle Mastery (500k-2M steps)
- `battle_win_rate` increases
- `battle_r` surpasses `exploration_r`
- Agent seeks + wins battles

### Phase 4: Milestone Progress (2M+ steps)
- `badges_earned` > 0
- `levels_gained` increases steadily
- Agent completes objectives

---

## Troubleshooting

### Problem: Agent still wanders
**Diagnosis**: Exploration rewards too high
**Fix**: Reduce `exploration_new_tile` by 50%

### Problem: Agent avoids battles
**Diagnosis**: Battle risk > reward
**Fix**: Increase `battle_start_bonus`, reduce `battle_loss` penalty

### Problem: Agent enters battles but always loses
**Diagnosis**: Battle mechanics not learned
**Fix**: Increase `battle_hp_delta` to reward partial success

### Problem: Agent gets stuck in corners
**Diagnosis**: Penalty too weak
**Fix**: Increase `penalty_wall` to -0.1 or -0.2

---

## Mathematical Justification

### Reward Signal Frequency

```
Exploration:  ~1.0 reward per step (dense)
Battle Start: ~0.002 reward per step (sparse, but high value)
Battle Win:   ~0.012 reward per step (sparse, very high value)
```

**Key Insight**: Even though battles are 500× less frequent, they must be 500× more rewarding to compete with exploration.

**Formula**:
```
battle_reward = exploration_reward × (steps_between_battles / steps_per_exploration)
              = 0.1 × (500 / 1)
              = 50.0 ✓
```

This is why `battle_win=50.0` and `exploration_new_tile=0.1` creates the right balance.

---

## Success Metrics

**Reward rebalancing is successful if**:
1. `reward_components/battle` > `reward_components/exploration` after 500k steps
2. `train/episode_battles_started_mean` > 1.0 after 1M steps
3. `train/episode_battles_won_mean` > 0.5 after 2M steps
4. Agent no longer hits max_steps without encountering battles

---

## References

- [PPO Paper](https://arxiv.org/abs/1707.06347) - Credit assignment in sparse reward environments
- [Reward Shaping](https://people.eecs.berkeley.edu/~pabbeel/cs287-fa09/readings/NgHaradaRussell-shaping-ICML1999.pdf) - Theoretical foundations
- Pokemon Red RAM Map - For milestone detection addresses
