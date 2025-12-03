# Curriculum Learning Guide - Pokemon Red RL

## Philosophy

**Progressive Task Difficulty**: Train agents on simpler subtasks before attempting the full game.

**Skill Isolation**: Separate navigation, battle, and integration skills where possible.

**Transfer Learning**: Later stages can initialize from earlier checkpoints.

---

## Curriculum Stages

### Stage 1: Pure Navigation (2-5M steps)

**Objective**: Learn to navigate to specific locations without battles.

**Config**: `configs/navigation_only.json`

**Key Settings**:
- `max_steps`: 2048
- `enable_battle`: false (battles disabled)
- `exploration_new_tile`: 1.0 (higher than redesigned configs)
- `milestone_key_location`: 50.0
- `termination_condition`: "pokecenter_reached"

**Expected Learning**:
- Agent learns arrow key controls
- Learns to avoid walls
- Finds path to target location
- **Success Metric**: Reaches Viridian Pokecenter in <2000 steps

**Training Command**:
```bash
python training/train_ppo.py \
  --config configs/navigation_only.json \
  --run-name nav_stage1 \
  --num-envs 12 \
  --total-multiplier 50 \
  --preset small
```

**What to Watch**:
- `train/episode_length_mean` should decrease
- `env_stats/coord_count` should stabilize (efficient path)
- `train/success_rate` should increase to >80%

---

### Stage 2: Battle Mechanics (10-20M steps)

**Objective**: Learn to find grass, enter battles, manage HP, and win.

**Config**: `configs/battle_focused_redesign.json`

**Key Settings**:
- `max_steps`: 4096
- `battle_start_bonus`: 15.0 (high reward for finding battles)
- `battle_win`: 60.0
- `exploration_new_tile`: 0.05 (low - discourage wandering)

**Expected Learning**:
- Agent seeks tall grass/trainers
- Learns battle mechanics (A button, move selection)
- Optimizes HP damage trade-offs
- **Success Metric**: `battles_started_mean` > 2.0, `win_rate` > 40%

**Training Command**:
```bash
python training/train_ppo.py \
  --config configs/battle_focused_redesign.json \
  --run-name battle_stage2 \
  --num-envs 12 \
  --total-multiplier 200 \
  --preset small
```

**What to Watch**:
- `train/episode_battles_started_mean` increases over time
- `train/episode_steps_to_first_battle` decreases
- `battle_stats/win_rate_mean` increases above 0.3

---

### Stage 3: Integrated Small-Map (20-50M steps)

**Objective**: Combine navigation + battles in constrained environment (Pallet + Route 1).

**Config**: `configs/balanced_redesign.json`

**Key Settings**:
- `max_steps`: 8192 (longer for combined tasks)
- Balanced rewards (exploration:battle:milestone = 1:50:500)
- All reward components enabled

**Expected Learning**:
- Navigate to grass → battle → level up → progress
- Strategic decision-making (grind vs advance)
- HP management + navigation
- **Success Metric**: `badges_earned` > 0 OR `levels_gained` > 5

**Optional Init State**: Custom save closer to Viridian/Route 1 grass

**Training Command**:
```bash
python training/train_ppo.py \
  --config configs/balanced_redesign.json \
  --run-name integrated_stage3 \
  --num-envs 12 \
  --total-multiplier 300 \
  --preset small
```

**What to Watch**:
- Both `exploration_r` and `battle_r` should be non-zero
- `train/episode_levels_gained_mean` > 1.0
- `train/episode_map_progress_max` increases

---

### Stage 4: Milestone Focus - First Badge (50-100M steps)

**Objective**: Reach Pewter Gym and defeat Brock for first badge.

**Config**: `configs/milestone_focused_redesign.json`

**Key Settings**:
- `max_steps`: 12288 (long episodes for complex objective)
- `milestone_badge`: 1000.0 (huge reward)
- `milestone_key_location`: 100.0 (reward for reaching gym)
- `termination_condition`: "badge_earned"

**Expected Learning**:
- Long-term planning (Pallet → Viridian → Forest → Pewter)
- Party management + grinding
- Gym leader strategy
- **Success Metric**: `badges_earned` = 1 within episode

**Training Command**:
```bash
python training/train_ppo.py \
  --config configs/milestone_focused_redesign.json \
  --run-name badge_stage4 \
  --num-envs 12 \
  --total-multiplier 400 \
  --preset medium
```

**What to Watch**:
- `train/episode_badges_earned_mean` > 0 (even 0.05 is progress)
- `train/success_rate` for badge termination
- `train/episode_map_progress_max` reaches Pewter Gym region

---

### Stage 5: Full Game (100M+ steps)

**Objective**: Complete the full Pokemon Red game (all 8 badges, Elite Four).

**Config**: `configs/full_game_shaped.json` (or create new with redesigned rewards)

**Key Settings**:
- `max_steps`: 40960+ (very long episodes)
- All reward components at balanced ratios
- No termination condition (or "elite_four_beaten")

**Expected Learning**:
- Full game strategy
- Resource management across long horizons
- Meta-learning (which Pokemon to catch, where to go)
- **Success Metric**: Multiple badges, deep game progression

**Training Command**:
```bash
python training/train_ppo.py \
  --config configs/full_game_redesigned.json \
  --run-name fullgame_stage5 \
  --num-envs 16 \
  --total-multiplier 2000 \
  --preset large
```

---

## Transfer Learning Between Stages

### Option 1: Curriculum as Separate Runs

Train each stage independently, document what works, manually advance to next stage.

**Pros**: Clear experimental control, easy to debug
**Cons**: No weight transfer

### Option 2: Initialize from Previous Checkpoint

Start Stage N from best checkpoint of Stage N-1.

**Command**:
```bash
# Stage 2 continues from Stage 1 best checkpoint
python training/train_ppo.py \
  --config configs/battle_focused_redesign.json \
  --run-name battle_stage2 \
  --resume-checkpoint runs/nav_stage1/poke_2000000_steps.zip \
  --num-envs 12
```

**Pros**: Faster convergence, leverages prior knowledge
**Cons**: May overfit to Stage 1 behavior

**Recommendation**: Try both, compare results.

---

## Curriculum Progression Checklist

### Before Moving to Next Stage

**From Stage 1 (Navigation) → Stage 2 (Battle)**:
- [ ] Agent reaches target in <2000 steps consistently
- [ ] `train/success_rate` > 0.8
- [ ] Exploration is efficient (not random wandering)

**From Stage 2 (Battle) → Stage 3 (Integrated)**:
- [ ] `battles_started_mean` > 1.5 per episode
- [ ] `battle_win_rate` > 0.3
- [ ] Agent actively seeks grass/trainers

**From Stage 3 (Integrated) → Stage 4 (Badge)**:
- [ ] `levels_gained_mean` > 3.0 per episode
- [ ] Both navigation and battle skills demonstrated
- [ ] Episode returns increasing steadily

**From Stage 4 (Badge) → Stage 5 (Full Game)**:
- [ ] At least 1 badge earned in evaluation
- [ ] `success_rate` > 0.1 for badge termination
- [ ] Long-term planning evident

---

## Troubleshooting Curriculum

### Problem: Agent doesn't transfer skills to next stage

**Diagnosis**: Task mismatch or catastrophic forgetting
**Solutions**:
1. Add small amount of Stage N-1 experience to Stage N training
2. Reduce learning rate when resuming (`--learning-rate 1e-4`)
3. Use curriculum annealing (gradually shift reward weights)

### Problem: Agent stuck in local optimum

**Diagnosis**: Exploration insufficient for new stage
**Solutions**:
1. Increase `ent_coef` temporarily (e.g., 0.02 → 0.05)
2. Add exploration bonus for new stage-specific events
3. Use epsilon-greedy policy during early Stage N training

### Problem: Skills learned in isolation don't combine

**Diagnosis**: Different policies for navigation vs battle
**Solutions**:
1. Add "integration bonus" - reward for navigation→battle→navigation chains
2. Ensure shared observation space (same features for both skills)
3. Consider multi-task learning (train on both simultaneously)

---

## Advanced: Battle-Only Isolated Environment (Future Work)

**Idea**: Create a modified environment that starts directly in battle state.

**Implementation** (requires code changes):
```python
class BattleOnlyEnv(RedGymEnv):
    def reset(self):
        super().reset()
        # Force battle state
        self.force_battle()  # Custom method to trigger battle
        return self._get_obs()

    def step(self, action):
        obs, reward, done, truncated, info = super().step(action)
        # Auto-reset to new battle when current one ends
        if not self.in_battle and not done:
            self.force_battle()
        return obs, reward, done, truncated, info
```

**Benefit**: True skill isolation - agent only learns battle mechanics
**Cost**: Significant implementation work, may not transfer well to full game

---

## Recommended Training Pipeline

### Week 1: Navigation Baseline
```bash
# Stage 1: Learn to navigate (2-3 days)
python training/train_ppo.py --config configs/navigation_only.json --run-name week1_nav --num-envs 12 --total-multiplier 100
```

### Week 2: Battle Training
```bash
# Stage 2: Learn battles (5-7 days)
python training/train_ppo.py --config configs/battle_focused_redesign.json --run-name week2_battles --num-envs 12 --total-multiplier 300
```

### Week 3: Integration
```bash
# Stage 3: Combine skills (7-10 days)
python training/train_ppo.py --config configs/balanced_redesign.json --run-name week3_integrated --num-envs 12 --total-multiplier 500
```

### Week 4+: Badge Rush
```bash
# Stage 4: First badge (10-14 days)
python training/train_ppo.py --config configs/milestone_focused_redesign.json --run-name week4_badge --num-envs 12 --total-multiplier 800
```

**Total**: ~1 month to first badge with current hardware/configs.

---

## Success Criteria

**Curriculum is successful if**:
1. Each stage shows clear learning (metrics improve over training)
2. Skills from earlier stages transfer to later stages
3. Agent reaches first badge within 100M total training steps
4. Final agent demonstrates strategic gameplay (not random actions)

**Metrics to Track Across All Stages**:
- Episode return (should increase)
- Success rate (stage-specific)
- Battle engagement (should be > 0 from Stage 2 onward)
- Efficiency (episode length should optimize for stage objective)

---

## Next Steps After Curriculum

1. **Hyperparameter tuning** for each stage
2. **Custom init states** to constrain training regions
3. **Auxiliary tasks** (predict next observation, etc.)
4. **Hierarchical policies** (high-level: where to go, low-level: how to navigate)
5. **Reward curriculum** (anneal coefficients over training)
