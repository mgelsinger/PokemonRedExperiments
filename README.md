# Pokemon Red RL Training

Train reinforcement learning agents to play Pokemon Red using PPO (Proximal Policy Optimization) with PyBoy emulation and Stable Baselines3.

<p float="left">
  <a href="https://youtu.be/DcYLT37ImBY">
    <img src="/assets/youtube.jpg?raw=true" height="192">
  </a>
  <a href="https://youtu.be/DcYLT37ImBY">
    <img src="/assets/poke_map.gif?raw=true" height="192">
  </a>
</p>

---

## Table of Contents

- [Features](#features)
- [Quick Start](#quick-start)
- [Training System](#training-system)
- [Curriculum Tasks](#curriculum-tasks)
- [Usage Examples](#usage-examples)
- [Documentation](#documentation)
- [Project Structure](#project-structure)
- [Requirements](#requirements)

---

## Features

### 🎯 Structured Reward Shaping
- **Exploration rewards**: New tile discovery, map progression
- **Battle rewards**: HP delta tracking, win/loss detection
- **Milestone rewards**: Badges, level-ups, key events
- **Penalty rewards**: Step penalties, wall collisions, stuck detection
- Fully configurable coefficients via JSON configs

### 📊 Rich TensorBoard Logging
- Reward component breakdowns (exploration, battle, milestone, penalty)
- Episode statistics (length, return)
- Battle statistics (wins, losses, win rate)
- PPO training metrics (KL divergence, clip fraction, entropy)
- Environment statistics (tiles visited, badges earned)
- Comprehensive histograms for all metrics

### 🎓 Curriculum Learning
- Progressive task difficulty from simple navigation to full game
- Task-specific termination conditions (badge earned, location reached)
- Pre-configured tasks optimized for different learning objectives
- 5 curriculum stages: navigation → exploration → battle → gym → full game

### 🛠️ Debug & Evaluation Tools
- **debug_rewards.py**: Validate reward shaping with random actions
- **eval_policy.py**: Evaluate trained checkpoints with detailed metrics
- **Comparison tools**: Side-by-side checkpoint comparison
- **Web dashboard**: Monitor training progress, manage runs

### ⚙️ Fully Configurable
- All PPO hyperparameters adjustable via JSON
- GPU memory presets (small, medium, large)
- Wandb integration for experiment tracking
- Resume training from checkpoints
- Customizable reward coefficients per task

---

## Quick Start

### Requirements
- **Python**: 3.10 or 3.11 (not 3.12+)
- **OS**: Windows with NVIDIA GPU + CUDA
- **Other**: ffmpeg on PATH

### Installation

**1. Install PyTorch with CUDA first:**
```bash
pip install torch==2.5.0+cu121 --index-url https://download.pytorch.org/whl/cu121
```

**2. Install dependencies:**
```bash
pip install -r requirements.txt
```

**3. Add Pokemon Red ROM:**
- Place your legally obtained `PokemonRed.gb` in the repo root
- Required SHA1: `ea9bcae617fdf159b045185467ae58b2e4a48b9a`
- `init.state` is already included for fast starts

### Verify Installation

Run the reward shaping test to ensure everything is working:
```bash
python tools/test_reward_shaping.py --rom PokemonRed.gb --state init.state
```

Expected output:
```
✓ All reward configurations loaded successfully
✓ Exploration rewards working correctly
✓ Battle reward system initialized
✓ Milestone reward system initialized
✓ Penalty rewards working correctly
✓ All tests passed!
```

---

## Training System

### Smoke Test (5-10 minutes)

Verify the training system works with a quick test:
```bash
python training/train_ppo.py \
  --config configs/walk_to_pokecenter.json \
  --run-name smoke_test \
  --total-multiplier 50 \
  --preset small
```

Then launch TensorBoard to view metrics:
```bash
tensorboard --logdir runs/smoke_test
```

### Full Training

Train on progressively harder curriculum tasks:

```bash
# 1. Navigation (500k steps, ~5-10 min)
python training/train_ppo.py \
  --config configs/walk_to_pokecenter.json \
  --run-name nav_01 \
  --total-multiplier 100 \
  --preset small

# 2. Exploration (10M steps, ~1 hour)
python training/train_ppo.py \
  --config configs/exploration_basic.json \
  --run-name explore_01 \
  --total-multiplier 1000 \
  --preset medium

# 3. Battle training (20M steps, ~2 hours)
python training/train_ppo.py \
  --config configs/battle_training.json \
  --run-name battle_01 \
  --total-multiplier 500 \
  --preset medium

# 4. Gym quest (40M steps, ~4 hours)
python training/train_ppo.py \
  --config configs/gym_quest.json \
  --run-name gym_01 \
  --total-multiplier 1000 \
  --preset large

# 5. Full game (100M+ steps, many hours)
python training/train_ppo.py \
  --config configs/full_game_shaped.json \
  --run-name full_game_01 \
  --total-multiplier 2000 \
  --preset large
```

### CLI Options

Common flags:
- `--config`: Path to task config JSON
- `--run-name`: Name for this training run
- `--total-multiplier`: Training duration (multiplier × 10k steps)
- `--preset`: GPU memory preset (small/medium/large)
- `--num-envs`: Number of parallel environments
- `--wandb`: Enable Weights & Biases logging
- `--no-stream`: Disable map streaming
- `--resume-latest`: Resume from latest checkpoint

See [docs/QUICK_REFERENCE.md](docs/QUICK_REFERENCE.md) for complete CLI reference.

---

## Curriculum Tasks

| Task | Objective | Termination | Reward Focus | Duration |
|------|-----------|-------------|--------------|----------|
| **walk_to_pokecenter** | Navigate to Viridian Pokecenter | Location reached | Exploration (10.0) | 500k-5M steps |
| **exploration_basic** | Explore Pallet Town & Route 1 | Max steps | Exploration (3.0) | 1M-10M steps |
| **battle_training** | Win battles efficiently | Max steps | Battle (5.0, 10.0 win) | 2M-20M steps |
| **gym_quest** | Earn first gym badge | Badge earned | Balanced | 4M-40M steps |
| **full_game_shaped** | Complete the full game | Max steps | All components | 10M-100M+ steps |

### Task-Specific Features

**walk_to_pokecenter** (Smoke Test Task):
- ✅ Simplest navigation objective
- ✅ Terminates when Pokecenter map (ID 40) is reached
- ✅ High exploration rewards (10.0 per new tile)
- ✅ Battles disabled (`enable_battle: false`)
- ✅ Ideal for validating training setup

**gym_quest**:
- ✅ Terminates when first badge is earned
- ✅ Balanced reward weights for all components
- ✅ Tests full game mechanics (exploration + battle + progression)

**battle_training**:
- ✅ High battle rewards (HP delta tracking, win bonuses)
- ✅ Lower exploration penalties to encourage battle seeking
- ✅ Tests combat optimization

All task configs in [`configs/`](configs/) directory.

---

## Usage Examples

### Train with Default Config
```bash
python training/train_ppo.py \
  --rom PokemonRed.gb \
  --state init.state \
  --run-name my_run
```

### Debug Reward Shaping
```bash
# Run random actions and see reward breakdown
python debug_rewards.py --task walk_to_pokecenter

# Multiple episodes with detailed output
python debug_rewards.py --task exploration_basic --episodes 5
```

### Evaluate Trained Policy
```bash
# Run 20 evaluation episodes
python eval_policy.py \
  --config configs/walk_to_pokecenter.json \
  --checkpoint runs/smoke_test/poke_500000_steps.zip \
  --n_episodes 20
```

Expected output:
```
=== Evaluation Results ===
Episodes: 20
Mean Return: 145.32 ± 23.45
Mean Length: 1234.5 ± 156.2
Success Rate: 75.0% (15/20)
```

### Play Checkpoint (Watch Agent)
```bash
python training/play_checkpoint.py \
  --checkpoint runs/my_run/poke_500000_steps.zip \
  --rom PokemonRed.gb \
  --state init.state
```

Add `--headless` to run without display, `--no-stream` to disable map streaming.

### Compare Two Checkpoints
```bash
python tools/compare_runs.py \
  --checkpoint-a runs/runA/poke_100000_steps.zip \
  --checkpoint-b runs/runB/poke_100000_steps.zip \
  --rom PokemonRed.gb \
  --state init.state
```

Generates side-by-side comparison PNG and summary JSON in `runs/compare_<timestamp>/`.

### Launch Web Dashboard
```bash
# Simple dashboard (list runs, copy commands)
python tools/serve_dashboard.py --port 8000

# Full control panel (start/stop runs, sliders)
python tools/ui_server.py
```

Then open http://localhost:8000 for web UI.

---

## Documentation

### 📚 Guides

- **[SMOKE_TEST_GUIDE.md](SMOKE_TEST_GUIDE.md)** - Step-by-step validation guide
  - Quick 5-10 minute test to verify everything works
  - What to check in TensorBoard
  - Troubleshooting common issues

- **[docs/TRAINING.md](docs/TRAINING.md)** - Complete training guide
  - TensorBoard metrics explanation
  - Hyperparameter tuning guide
  - Curriculum progression strategies
  - Advanced configuration options

- **[docs/DEBUG_AND_EVAL_GUIDE.md](docs/DEBUG_AND_EVAL_GUIDE.md)** - Debug & evaluation tools
  - How to use `debug_rewards.py`
  - How to use `eval_policy.py`
  - Reward shaping validation
  - Policy evaluation workflows

- **[docs/QUICK_REFERENCE.md](docs/QUICK_REFERENCE.md)** - Command reference
  - All CLI flags and options
  - Quick copy-paste commands
  - Common use cases

### 📋 Implementation Details

- **[IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md)** - Feature implementation details
  - What was implemented
  - File-by-file changes
  - Code structure overview

- **[IMPLEMENTATION_VALIDATION.md](IMPLEMENTATION_VALIDATION.md)** - Validation report
  - Code verification results
  - Implementation completeness (10/12 features)
  - Environment setup notes

- **[VERIFICATION_AUDIT.md](VERIFICATION_AUDIT.md)** - Line-by-line audit
  - Detailed implementation status
  - Code references with line numbers

---

## Project Structure

```
├── env/
│   ├── red_gym_env.py          # Main Gym environment
│   ├── reward_config.py        # Reward configuration system
│   └── map_data.json           # Pokemon Red map metadata
├── training/
│   ├── train_ppo.py            # PPO training script
│   ├── play_checkpoint.py      # Play trained checkpoints
│   └── tensorboard_callback.py # Custom TensorBoard logging
├── configs/
│   ├── walk_to_pokecenter.json # Navigation task (smoke test)
│   ├── exploration_basic.json  # Exploration task
│   ├── battle_training.json    # Battle task
│   ├── gym_quest.json          # First gym badge task
│   ├── full_game_shaped.json   # Full game task
│   └── train_default.json      # Default training config
├── tools/
│   ├── test_reward_shaping.py  # Reward validation test
│   ├── smoke_test.py           # Quick sanity check
│   ├── compare_runs.py         # Compare two checkpoints
│   ├── serve_dashboard.py      # Simple web dashboard
│   └── ui_server.py            # Full control panel
├── docs/
│   ├── TRAINING.md             # Complete training guide
│   ├── DEBUG_AND_EVAL_GUIDE.md # Debug/eval tools guide
│   └── QUICK_REFERENCE.md      # Command reference
├── debug_rewards.py            # Debug reward shaping tool
├── eval_policy.py              # Evaluate trained policies
├── SMOKE_TEST_GUIDE.md         # Quick test guide
├── PokemonRed.gb               # Pokemon Red ROM (you provide)
└── init.state                  # Initial save state (included)
```

---

## TensorBoard Metrics

When you run training, TensorBoard logs detailed metrics organized into tabs:

### Reward Components (`reward_components/`)
- `exploration` - Reward from discovering new tiles
- `battle` - Reward from battles (HP delta, wins/losses)
- `milestone` - Reward from badges, levels, events
- `penalty` - Penalties (steps, walls, stuck)
- `total_shaped` - Sum of all components
- `legacy` - Original reward (for comparison)

### Battle Statistics (`battle_stats/`)
- `wins_mean` - Average battles won per episode
- `losses_mean` - Average battles lost per episode
- `total_mean` - Average total battles per episode
- `win_rate_mean` - Win rate percentage

### Episode Statistics (`episode/`)
- `length_mean` - Average episode length
- `length_max` - Maximum episode length
- `length_min` - Minimum episode length
- `length_distrib` - Episode length histogram

### PPO Training (`train/`)
- `approx_kl` - KL divergence (should be small, <0.1)
- `clip_fraction` - Fraction of clipped updates (0.1-0.3 ideal)
- `entropy_loss` - Policy entropy (should decrease slowly)
- `explained_variance` - Value function quality (→1.0 is good)
- `learning_rate` - Current learning rate

### Environment Stats (`env_stats/`)
- `coord_count` - Unique tiles visited
- `badge` - Number of badges earned
- `max_map_progress` - Map progression metric

See [docs/TRAINING.md](docs/TRAINING.md) for detailed metrics explanation.

---

## Advanced Features

### Resume Training
```bash
# Resume from latest checkpoint
python training/train_ppo.py \
  --config configs/gym_quest.json \
  --run-name gym_01 \
  --resume-latest

# Resume from specific checkpoint
python training/train_ppo.py \
  --config configs/gym_quest.json \
  --run-name gym_01 \
  --resume-checkpoint runs/gym_01/poke_10000000_steps.zip
```

### Wandb Integration
```bash
python training/train_ppo.py \
  --config configs/exploration_basic.json \
  --run-name explore_wandb \
  --wandb
```

### Custom Hyperparameters
```bash
# Override config values via CLI
python training/train_ppo.py \
  --config configs/gym_quest.json \
  --run-name custom_lr \
  --learning-rate 0.0001 \
  --batch-size 1024 \
  --num-envs 32
```

### Evaluation During Training
```bash
python training/train_ppo.py \
  --config configs/gym_quest.json \
  --run-name gym_eval \
  --eval-every-steps 100000 \
  --eval-episodes 10 \
  --eval-log runs/gym_eval/eval.jsonl
```

---

## Troubleshooting

### Python Version Issues

**Problem**: `pyboy` fails to compile on Python 3.12+

**Solution**: Use Python 3.10 or 3.11
```bash
# Using conda
conda create -n pokemon python=3.11
conda activate pokemon
pip install -r requirements.txt

# Using pyenv
pyenv install 3.11.9
pyenv local 3.11.9
pip install -r requirements.txt
```

### GPU/CUDA Issues

**Problem**: Training doesn't use GPU

**Solution**: Verify CUDA is available
```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

If False, reinstall PyTorch with CUDA:
```bash
pip uninstall torch
pip install torch==2.5.0+cu121 --index-url https://download.pytorch.org/whl/cu121
```

### TensorBoard Shows No Metrics

**Problem**: TensorBoard doesn't show `reward_components/` or `battle_stats/`

**Solution**:
- Wait for first episode to complete (~40k steps for walk_to_pokecenter)
- Refresh browser (Ctrl+R)
- Check correct logdir: `tensorboard --logdir runs/<your_run_name>`

### Training Crashes Immediately

**Problem**: Out of memory or other startup error

**Solution**:
- Use `--preset small` to reduce memory usage
- Reduce `--num-envs` (try 8 or 4)
- Check ROM file exists and matches SHA1

See [SMOKE_TEST_GUIDE.md](SMOKE_TEST_GUIDE.md) for more troubleshooting.

---

## Implementation Status

### ✅ Fully Implemented (10/12)

1. **Reward shaping structure** - Separated components with tracking
2. **Configurable coefficients** - RewardConfig dataclass + JSON
3. **Exploration rewards** - Episode-specific tile tracking
4. **Battle rewards** - HP delta, win/loss detection
5. **Battle statistics** - Win/loss tracking + TensorBoard metrics
6. **Milestone rewards** - Badges, levels, events, locations
7. **Penalty rewards** - Step, wall, stuck penalties
8. **TensorBoard logging** - All components + histograms
9. **Curriculum tasks** - 5 progressive tasks
10. **Termination conditions** - badge_earned, pokecenter_reached
11. **PPO hyperparameters** - Fully configurable
12. **Debug/eval tools** - debug_rewards.py, eval_policy.py

### ⚠️ Future Enhancements (2/12)

1. **Entropy schedule** - Currently static `ent_coef`
2. **Task-specific states** - All tasks use same `init.state`

---

## Credits

- **PyBoy**: Game Boy emulator - https://github.com/Baekalfen/PyBoy
- **Stable-Baselines3**: RL algorithms - https://github.com/DLR-RM/stable-baselines3
- **Original Pokemon Red**: Nintendo/Game Freak

---

## License

This project is for educational and research purposes. Pokemon Red is property of Nintendo/Game Freak. Please ensure you own a legal copy of the game.

---

## Getting Help

- **Quick Test**: [SMOKE_TEST_GUIDE.md](SMOKE_TEST_GUIDE.md)
- **Training Guide**: [docs/TRAINING.md](docs/TRAINING.md)
- **Debug/Eval**: [docs/DEBUG_AND_EVAL_GUIDE.md](docs/DEBUG_AND_EVAL_GUIDE.md)
- **Commands**: [docs/QUICK_REFERENCE.md](docs/QUICK_REFERENCE.md)

For bugs or questions, please open an issue on GitHub.
