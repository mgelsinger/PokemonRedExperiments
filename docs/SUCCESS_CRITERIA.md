# Success Criteria for Pokemon Red RL Agent

This document defines **quantitative success metrics** for determining whether the RL agent is making "meaningful progress" at each curriculum stage. Use these criteria to validate training runs and diagnose issues.

---

## Overview

The goal is to move from an agent that **wanders randomly** to one that can:
1. Navigate to specific locations
2. Seek out and win wild battles
3. Earn gym badges
4. Make consistent progress through the game

Each curriculum stage has specific success criteria that must be met before advancing to the next stage.

---

## Stage 1: Navigation Only

**Config**: `configs/navigation_only.json`

**Training Duration**: 2-10M steps

### Success Metrics

| Metric | Target | Description |
|--------|--------|-------------|
| `eval/mean_reward` | > 50 | Agent discovers at least 50 new tiles per episode |
| `eval/mean_length` | < 2048 | Agent doesn't waste steps in loops |
| `trajectory/explore_sum` | > 500 unique tiles | Exploration map shows diverse coverage |
| `train/exploration_return` | > 100 | Consistent exploration rewards during training |
| `train/episode_map_progress_max` | > 1 | Agent reaches beyond starting area (map_id > 0) |

### Validation Commands

```bash
# Train navigation-only stage
python training/train_ppo.py --config configs/navigation_only.json \
    --total-steps 10000000 \
    --eval-every-steps 500000 \
    --num-envs 12

# Check if exploration is working
tensorboard --logdir logs/
# Look for: eval/mean_reward increasing, trajectory/explore_sum filling in
```

### Expected Behavior

- **Early (0-2M steps)**: Random movement, ~10-20 mean reward
- **Mid (2-5M steps)**: Agent learns to avoid walls, ~30-50 mean reward
- **Late (5-10M steps)**: Systematic exploration, >50 mean reward, visits Viridian City, Route 2

### Red Flags

- ❌ Mean reward stuck at <10: Agent is stuck in loops or hitting walls
- ❌ Explore map shows only 1-2 rooms: No navigation progress
- ❌ Episode length always hits max_steps: Agent never terminates naturally

---

## Stage 2: Battle-Focused Training

**Config**: `configs/battle_focused_redesign.json`

**Training Duration**: 10-50M steps

### Success Metrics

| Metric | Target | Description |
|--------|--------|-------------|
| `eval/mean_battles_started` | > 0.5 | Agent encounters battles in >50% of eval episodes |
| `eval/mean_battles_won` | > 0.3 | Agent wins at least 30% of discovered battles |
| `train/episode_battles_started_mean` | > 1.0 | Multiple battles per training episode |
| `train/episode_steps_to_first_battle` | < 500 | Agent finds battles quickly |
| `battle_stats/win_rate_mean` | > 0.4 | Win rate above 40% indicates learning |
| `eval/mean_reward` | > 100 | Battle rewards (50-60 per win) dominate |

### Validation Commands

```bash
# Train battle-focused stage
python training/train_ppo.py --config configs/battle_focused_redesign.json \
    --total-steps 50000000 \
    --eval-every-steps 1000000 \
    --num-envs 12

# Monitor battle discovery
tensorboard --logdir logs/
# Look for: train/episode_battles_started_mean, battle_stats/win_rate_mean
```

### Expected Behavior

- **Early (0-10M steps)**: Random battles, win_rate ~0.2-0.3
- **Mid (10-30M steps)**: Agent learns to run from losing battles, win_rate ~0.4
- **Late (30-50M steps)**: Agent seeks tall grass, win_rate >0.5, levels up starter Pokemon

### Red Flags

- ❌ `battles_started_mean` stuck at 0: Agent never enters tall grass
- ❌ `steps_to_first_battle` > 2000: Agent avoiding battles or stuck
- ❌ `win_rate` < 0.2: Agent not learning battle strategy (run from losing fights)

---

## Stage 3: Balanced Exploration + Battles

**Config**: `configs/balanced_redesign.json`

**Training Duration**: 20-100M steps

### Success Metrics

| Metric | Target | Description |
|--------|--------|-------------|
| `eval/mean_battles_started` | > 1.0 | Multiple battles per episode |
| `eval/mean_battles_won` | > 0.7 | High win rate from trained starter |
| `eval/mean_levels_gained` | > 2.0 | Starter reaches level 7+ |
| `train/milestone_return` | > 50 | Leveling rewards accumulating |
| `trajectory/explore_sum` | > 1000 tiles | Explores beyond Viridian Forest |
| `train/episode_map_progress_max` | > 3 | Reaches Pewter City area |

### Validation Commands

```bash
# Train balanced stage
python training/train_ppo.py --config configs/balanced_redesign.json \
    --total-steps 100000000 \
    --eval-every-steps 2000000 \
    --num-envs 12

# Evaluate checkpoint manually
python training/evaluate.py --model logs/balanced_redesign_100M/model.zip \
    --episodes 50 --render
```

### Expected Behavior

- **Early (0-20M steps)**: Balances exploration and battles, ~2-3 battles per episode
- **Mid (20-60M steps)**: Starter reaches level 7-9, begins winning consistently
- **Late (60-100M steps)**: Agent navigates Viridian Forest, defeats trainers, reaches Pewter City

### Red Flags

- ❌ `levels_gained` stuck at 0: Not winning enough battles to level up
- ❌ `map_progress_max` stuck at 0-1: Never leaves Pallet/Viridian area
- ❌ `mean_battles_won` < `mean_battles_started`: Losing most battles (need better strategy)

---

## Stage 4: Milestone-Focused (Badge Rush)

**Config**: `configs/milestone_focused_redesign.json`

**Training Duration**: 50-200M steps

### Success Metrics

| Metric | Target | Description |
|--------|--------|-------------|
| `eval/mean_badges_earned` | > 0.1 | At least 10% of eval episodes earn Boulder Badge |
| `eval/mean_levels_gained` | > 5.0 | Starter reaches level 12+ for Brock fight |
| `eval/success_rate` | > 0.1 | 10%+ eval episodes complete objective |
| `train/episode_badges_earned_mean` | > 0.05 | Some training episodes earn badges |
| `battle_stats/win_rate_mean` | > 0.7 | Consistently winning battles |
| `trajectory/all_flags` | Contains "BEAT_BROCK" | Event flag set for defeating gym leader |

### Validation Commands

```bash
# Train milestone stage
python training/train_ppo.py --config configs/milestone_focused_redesign.json \
    --total-steps 200000000 \
    --eval-every-steps 5000000 \
    --num-envs 12 \
    --resume logs/balanced_redesign_100M/model.zip  # Transfer learning

# Check for badge progress
grep "badges_earned" logs/milestone_focused_redesign/eval.jsonl | tail -20
```

### Expected Behavior

- **Early (0-50M steps)**: Agent navigates to Pewter Gym, explores inside
- **Mid (50-120M steps)**: Agent fights gym trainers, levels starter to 12+
- **Late (120-200M steps)**: Agent defeats Brock (if starter type advantage), earns Boulder Badge

### Red Flags

- ❌ `badges_earned` always 0 after 100M+ steps: Not reaching Brock or losing fight
- ❌ `deaths_mean` > 5: Agent dying repeatedly in gym
- ❌ Starter stuck at level 7-9: Not battling enough before gym challenge

---

## Universal Red Flags (All Stages)

These indicate fundamental training issues:

| Symptom | Diagnosis | Fix |
|---------|-----------|-----|
| `train/approx_kl` > 0.1 | Policy updates too aggressive | Reduce `learning_rate` or `clip_range` |
| `train/entropy_loss` → 0 early | Policy collapsed to deterministic | Increase `ent_coef` |
| `rollout/ep_rew_mean` negative | Penalties dominating rewards | Reduce `penalty_step` or increase reward scale |
| GPU util < 50% | CPU bottleneck from RAM thrashing | Reduce `num_envs` (RAM limit hit) |
| `train/loss` not decreasing | Learning not happening | Check reward signal, increase `learning_rate` |
| `train/value_loss` >> `policy_gradient_loss` | Value function struggling | Increase `vf_coef` or `n_epochs` |

---

## Testing Protocol

### Quick Smoke Test (2-3 hours)

```bash
# Test navigation (2M steps, ~30 min)
python training/train_ppo.py --config configs/navigation_only.json \
    --total-steps 2000000 --num-envs 12

# Test battle focus (10M steps, ~2 hours)
python training/train_ppo.py --config configs/battle_focused_redesign.json \
    --total-steps 10000000 --num-envs 12 \
    --eval-every-steps 1000000
```

**Success**: navigation eval_reward > 30, battle battles_started > 0.3

### Full Curriculum Test (2-3 days)

```bash
# Stage 1: Navigation (10M steps, ~4 hours)
python training/train_ppo.py --config configs/navigation_only.json \
    --total-steps 10000000 --num-envs 12 --name stage1_nav

# Stage 2: Battles (50M steps, ~12 hours)
python training/train_ppo.py --config configs/battle_focused_redesign.json \
    --total-steps 50000000 --num-envs 12 --name stage2_battle \
    --resume logs/stage1_nav/model_10000000_steps.zip

# Stage 3: Balanced (100M steps, ~24 hours)
python training/train_ppo.py --config configs/balanced_redesign.json \
    --total-steps 100000000 --num-envs 12 --name stage3_balanced \
    --resume logs/stage2_battle/model_50000000_steps.zip

# Stage 4: Milestone (200M steps, ~48 hours)
python training/train_ppo.py --config configs/milestone_focused_redesign.json \
    --total-steps 200000000 --num-envs 12 --name stage4_milestone \
    --resume logs/stage3_balanced/model_100000000_steps.zip
```

**Success**: Final model earns Boulder Badge in >10% of eval episodes

---

## Debugging Guide

### Agent Not Exploring

**Symptoms**: `eval/mean_reward` stuck at <10, explore_sum only shows 1-2 rooms

**Diagnosis**:
1. Check TensorBoard for `train/penalty_return` - are wall penalties dominating?
2. Check `trajectory/explore_map` - is agent stuck in a small loop?
3. Check `train/entropy_loss` - has policy become deterministic too early?

**Fixes**:
- Increase `ent_coef` from 0.01 to 0.02 (more randomness)
- Reduce `penalty_wall` from -0.05 to -0.01 (less punishment)
- Increase `exploration_new_tile` reward (more encouragement)

### Agent Not Finding Battles

**Symptoms**: `battles_started_mean` stuck at 0 after 10M+ steps

**Diagnosis**:
1. Check `trajectory/explore_sum` - does agent explore tall grass areas?
2. Check `train/exploration_return` - is exploration reward too high? (agent prefers safe tiles)
3. Check `battle_start_bonus` - is it high enough to encourage tall grass?

**Fixes**:
- Increase `battle_start_bonus` from 10.0 to 20.0
- Reduce `exploration_new_tile` from 0.1 to 0.05 (make battles more attractive)
- Check if RNG seed is preventing battles (bad luck in eval)

### Agent Not Winning Battles

**Symptoms**: `win_rate < 0.3` even after 20M+ steps

**Diagnosis**:
1. Check `train/episode_levels_gained` - is starter leveling up?
2. Check battle strategy - is agent selecting good moves or running?
3. Check `battle_loss` penalty - is it too harsh (-10.0)?

**Fixes**:
- Reduce `battle_loss` penalty from -10.0 to -5.0 (less punishment for losing)
- Increase `battle_hp_delta` reward (encourage dealing damage even if losing)
- Manually verify battle mechanics are working (check RAM reads)

### Agent Not Earning Badges

**Symptoms**: `badges_earned` always 0 after 100M+ steps in milestone stage

**Diagnosis**:
1. Check `map_progress_max` - is agent reaching Pewter City (map_id ~3)?
2. Check `levels_gained` - is starter reaching level 12+ before Brock?
3. Check `deaths_mean` - is agent dying in gym before reaching Brock?

**Fixes**:
- Increase episode length (`max_steps` to 16384 for more time)
- Pre-train on navigation + battles before milestone stage
- Increase `milestone_badge` reward from 500 to 1000 (more motivation)
- Check if game state is correctly detecting badge acquisition

---

## Expected Training Curves

### Navigation Stage (0-10M steps)

```
eval/mean_reward:      [5 → 30 → 60]
explore_sum:           [50 → 300 → 800]
map_progress_max:      [0 → 1 → 2]
```

### Battle Stage (0-50M steps)

```
battles_started_mean:  [0.1 → 0.8 → 2.0]
win_rate:              [0.2 → 0.4 → 0.6]
levels_gained:         [0 → 1.5 → 4.0]
steps_to_first_battle: [1500 → 600 → 300]
```

### Balanced Stage (0-100M steps)

```
battles_won:           [0.5 → 2.0 → 4.0]
levels_gained:         [1 → 4 → 8]
map_progress_max:      [1 → 2 → 4]
mean_reward:           [50 → 200 → 400]
```

### Milestone Stage (0-200M steps)

```
badges_earned:         [0 → 0.02 → 0.15]
levels_gained:         [3 → 8 → 12]
success_rate:          [0 → 0.05 → 0.2]
```

---

## Final Checklist: "Is My Agent Learning?"

Use this checklist after 50M steps on any config:

- [ ] `eval/mean_reward` is increasing over time (not flat)
- [ ] `train/approx_kl` stays below 0.05 (stable learning)
- [ ] `train/loss` is decreasing (optimization working)
- [ ] `battle_stats/win_rate` > 0.3 (battle strategy emerging)
- [ ] `trajectory/explore_sum` shows >500 unique tiles
- [ ] `train/episode_battles_started_mean` > 0.5 (battle discovery)
- [ ] `eval/mean_levels_gained` > 1.0 (winning enough to level up)
- [ ] TensorBoard shows smooth curves (not erratic/noisy)
- [ ] GPU utilization >80% (not CPU-bound from RAM issues)
- [ ] `episode/length_mean` increasing over time (surviving longer)

**If 7+ boxes checked**: Agent is learning meaningfully ✅
**If 3-6 boxes checked**: Agent is learning but needs tuning ⚠️
**If <3 boxes checked**: Training is failing, see Debugging Guide ❌

---

## Hardware-Specific Success Criteria

### For User's Hardware (RTX 4080, 32GB RAM, i9-13900K)

**Recommended Configuration**:
- `num_envs`: 12 (RAM constraint)
- `rollout_horizon`: 512 (RAM constraint)
- `batch_size`: 1024 (fits in 16GB VRAM)
- Training speed: ~250-300 steps/sec

**Realistic Milestones**:
- 10M steps: ~9 hours
- 50M steps: ~46 hours (2 days)
- 100M steps: ~92 hours (4 days)
- 200M steps: ~185 hours (8 days)

**Success Adjusted for Hardware**:

Due to RAM constraints (12 envs instead of ideal 32-64), learning will be **slower**:
- Navigation: 10M → **20M steps** to reach targets
- Battles: 50M → **100M steps** to reach win_rate > 0.5
- Balanced: 100M → **200M steps** to reach Pewter City
- Milestone: 200M → **400M+ steps** to earn first badge

**Strategy**: Use longer training runs and lower expectations for early milestones.

---

## Next Steps After Success

Once the agent can earn the Boulder Badge (Stage 4 success):

1. **Extend Milestone Config**: Add more badges (Cascade, Thunder, etc.)
2. **Implement State Initialization**: Load game saves near gyms to accelerate training
3. **Add Curriculum Annealing**: Gradually reduce reward shaping, rely more on intrinsic progress
4. **Optimize Memory**: Implement agent_stats sampling to enable 32+ envs
5. **Transfer Learning**: Fine-tune from badge checkpoint for full game completion

See [docs/CURRICULUM_GUIDE.md](CURRICULUM_GUIDE.md) for extended curriculum design.

---

## Summary

**Minimum Viable Success** (After 100M steps on balanced config):
- Agent can navigate from Pallet Town → Viridian City → Viridian Forest → Pewter City
- Agent initiates >1 battle per episode and wins >50% of them
- Agent levels starter to 7+ through battle wins
- Agent discovers >1000 unique tiles

**Stretch Goal** (After 400M steps on milestone config):
- Agent earns Boulder Badge in >10% of evaluation episodes
- Agent demonstrates strategic battle behavior (running from losing fights)
- Agent navigates complex multi-room dungeons (Viridian Forest)
- Metrics show consistent improvement over final 100M steps

Use the metrics above to determine when to advance between curriculum stages or when to diagnose training failures.
