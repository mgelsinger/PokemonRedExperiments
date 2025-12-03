# Smoke Test Execution Guide

## Quick Verification Steps

Follow these steps to verify the improved training system is working correctly.

---

## Step 1: Verify Reward Shaping

**Time**: 1-2 minutes

```bash
python tools/test_reward_shaping.py --rom PokemonRed.gb --state init.state
```

**Expected Output**:
- ✓ All reward configurations loaded successfully
- ✓ Exploration rewards working correctly
- ✓ Battle reward system initialized
- ✓ Milestone reward system initialized
- ✓ Penalty rewards working correctly
- ✓ All tests passed!

**If this fails**: There's a configuration or code issue. Check error messages.

---

## Step 2: Run Smoke Test Training

**Time**: 5-10 minutes

**Recommended command** (shortest test):
```bash
python training/train_ppo.py \
  --config configs/walk_to_pokecenter.json \
  --run-name smoke_test \
  --total-multiplier 50 \
  --preset small
```

This runs approximately **500,000 steps** which should take 5-10 minutes.

**Alternative** (slightly longer, 1M steps):
```bash
python training/train_ppo.py \
  --config configs/walk_to_pokecenter.json \
  --run-name smoke_test \
  --total-multiplier 100 \
  --preset small
```

**Alternative** (if you have more time, 2M steps):
```bash
python training/train_ppo.py \
  --config configs/exploration_basic.json \
  --run-name smoke_exploration \
  --total-multiplier 200 \
  --preset medium
```

**Expected Output (during training)**:
- Model policy architecture printed
- Regular checkpoint saves
- Progress updates every few thousand steps
- No crashes or errors

---

## Step 3: Launch TensorBoard

**Open a new terminal window** and run:

```bash
tensorboard --logdir runs/smoke_test
```

Then open your browser to: `http://localhost:6006`

---

## Step 4: Verify TensorBoard Metrics

Navigate through TensorBoard tabs and verify these metrics appear:

### ✅ Reward Components Tab (`reward_components/`)

**Must be present and non-zero**:
- `exploration` - Should be **positive and increasing** (agent discovering new tiles)
- `penalty` - Should be **negative** (small step penalties)
- `total_shaped` - Should show **overall episode return**
- `milestone` - May be low/zero for walk_to_pokecenter task
- `battle` - Should be **zero** for walk_to_pokecenter (no battles)
- `legacy` - Legacy rewards for comparison

**What to check**:
- [ ] All metrics appear in the list
- [ ] Exploration reward is positive and > 0
- [ ] Total shaped reward is being logged
- [ ] Graphs are updating as training progresses

### ✅ Episode Statistics Tab (`episode/`)

**Must be present**:
- `length_mean` - Average episode length
- `length_max` - Maximum episode length
- `length_min` - Minimum episode length

**What to check**:
- [ ] Episode length is reasonable (not stuck at max_steps always)
- [ ] Length varies over time (some episodes shorter than others)
- [ ] Metrics update regularly

### ✅ Battle Statistics Tab (`battle_stats/`)

**For walk_to_pokecenter task** (no battles expected):
- `wins_mean` - Should be **0**
- `losses_mean` - Should be **0**
- `total_mean` - Should be **0**
- `win_rate_mean` - Should be **0**

**What to check**:
- [ ] Metrics appear (even if zero)
- [ ] No errors in TensorBoard

**Note**: For battle_training task, these should be **non-zero**.

### ✅ PPO Training Metrics Tab (`train/`)

**Must be present**:
- `approx_kl` - Should be **small** (0.01-0.05 is good)
- `clip_fraction` - Should be **0.1-0.3**
- `entropy_loss` - Should **slowly decrease**
- `explained_variance` - Should **approach 1.0**
- `learning_rate` - Should match config (0.0005 for walk_to_pokecenter)

**What to check**:
- [ ] All metrics present
- [ ] approx_kl is not too large (>0.5 = problem)
- [ ] clip_fraction is reasonable
- [ ] entropy_loss decreases over time

### ✅ Environment Stats Tab (`env_stats/`)

**Must be present**:
- `coord_count` - Number of unique tiles visited
- `badge` - Badge count (0 for walk_to_pokecenter)
- `max_map_progress` - Map progression

**What to check**:
- [ ] coord_count increases over time
- [ ] Metrics update regularly

### ✅ Histograms Tab

**Must be present**:
- `reward_distribs/` - Histograms for all reward components
- `battle_stats/` - Histograms for battle wins/losses
- `episode/length_distrib` - Episode length distribution

**What to check**:
- [ ] Histograms are visible
- [ ] Distributions update over time

---

## Step 5: Verify Checkpoints

Check that checkpoints were saved:

```bash
# Windows PowerShell
ls runs/smoke_test/*.zip

# Or Windows CMD
dir runs\smoke_test\*.zip
```

**Expected Output**:
- At least one `.zip` file (e.g., `poke_250000_steps.zip`)
- Files named like `poke_<steps>_steps.zip`

**What to check**:
- [ ] At least one checkpoint file exists
- [ ] Checkpoints have reasonable file sizes (>1MB)

---

## Step 6: Test Playback (Optional)

If you want to see what the trained agent does:

```bash
python training/play_checkpoint.py \
  --checkpoint runs/smoke_test/poke_<XXXXX>_steps.zip \
  --rom PokemonRed.gb \
  --state init.state
```

Replace `<XXXXX>` with the actual step number from your checkpoint.

This will open a window showing the agent playing (if not headless).

---

## Success Criteria Checklist

Your smoke test **PASSES** if:

- [ ] ✅ test_reward_shaping.py completes without errors
- [ ] ✅ Training runs for at least 500k steps without crashing
- [ ] ✅ TensorBoard shows `reward_components/exploration` > 0
- [ ] ✅ TensorBoard shows `reward_components/penalty` < 0
- [ ] ✅ TensorBoard shows `reward_components/total_shaped` is being logged
- [ ] ✅ TensorBoard shows `battle_stats/*` metrics (even if zero)
- [ ] ✅ TensorBoard shows `episode/length_mean`
- [ ] ✅ TensorBoard shows `train/approx_kl` is small (<0.1)
- [ ] ✅ At least one checkpoint file was saved
- [ ] ✅ No Python errors or crashes during training

---

## Troubleshooting

### Problem: test_reward_shaping.py fails

**Solution**: Check the error message. Likely issues:
- ROM or state file not found
- Python dependency missing
- Syntax error (shouldn't happen if code was not modified)

### Problem: Training crashes immediately

**Solution**:
- Check that ROM and state files exist
- Verify GPU/CUDA is working: `python -c "import torch; print(torch.cuda.is_available())"`
- Try `--preset small` to reduce memory usage
- Check error messages

### Problem: TensorBoard metrics are all zero

**Solution**:
- Wait longer - metrics update after first episode completes
- Episode length is `max_steps` (40960 for walk_to_pokecenter)
- First metrics appear after ~40k steps

### Problem: TensorBoard doesn't show reward_components/

**Solution**:
- Make sure you're looking at the correct run: `runs/smoke_test`
- Refresh TensorBoard (Ctrl+R in browser)
- Check that training is actually running
- Metrics appear after first episode completes

### Problem: approx_kl is very large (>0.5)

**Solution**:
- Decrease learning_rate in config
- This usually stabilizes after a few updates
- If persistent, reduce `clip_range`

### Problem: explained_variance is negative

**Solution**:
- This is normal at the very start of training
- Should improve after a few thousand steps
- If stays negative: value function not learning well
- Try increasing `vf_coef` or `n_epochs`

---

## Next Steps After Smoke Test Passes

1. **Run longer training** on walk_to_pokecenter:
   ```bash
   python training/train_ppo.py \
     --config configs/walk_to_pokecenter.json \
     --run-name walk_long \
     --total-multiplier 500 \
     --preset medium
   ```

2. **Try other curriculum tasks**:
   ```bash
   # Exploration (10M steps, ~1 hour)
   python training/train_ppo.py \
     --config configs/exploration_basic.json \
     --run-name exploration_01 \
     --total-multiplier 1000

   # Battle training (20M steps, ~2 hours)
   python training/train_ppo.py \
     --config configs/battle_training.json \
     --run-name battle_01 \
     --total-multiplier 500

   # Gym quest (40M steps, ~4 hours)
   python training/train_ppo.py \
     --config configs/gym_quest.json \
     --run-name gym_01 \
     --total-multiplier 1000
   ```

3. **Monitor and iterate**:
   - Watch TensorBoard metrics
   - Adjust reward coefficients if needed
   - Try different hyperparameters
   - Experiment with different tasks

---

## Reference

- **Full training guide**: [docs/TRAINING.md](docs/TRAINING.md)
- **Quick reference**: [docs/QUICK_REFERENCE.md](docs/QUICK_REFERENCE.md)
- **Implementation details**: [IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md)
- **Audit report**: [AUDIT_REPORT.md](AUDIT_REPORT.md)
