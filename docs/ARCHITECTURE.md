# Architecture Documentation - Pokemon Red RL Training

**Generated**: Phase 1 - Repo and Environment Recon
**Purpose**: Comprehensive architecture overview for RL improvement planning

---

## 1. Environment Pipeline

### Emulator → Wrappers → RL Environment

```
PyBoy Emulator (PokemonRed.gb)
    ↓
RedGymEnv (env/red_gym_env.py)
    - Wraps PyBoy with Gym interface
    - Manages game state reading (RAM addresses)
    - Implements reward shaping
    - Tracks episode statistics
    ↓
StreamWrapper (optional, env/stream_agent_wrapper.py)
    - Adds map visualization overlay
    - Sends state to WebSocket server
    ↓
SubprocVecEnv (Stable-Baselines3)
    - Parallelizes N environment instances
    ↓
VecMonitor (Stable-Baselines3)
    - Adds episode-level metrics logging
    ↓
PPO Agent (training/train_ppo.py)
```

### Key Components

**RedGymEnv** (`env/red_gym_env.py`):
- **Observation Space** (Dict):
  - `screens`: (72, 80, 3) uint8 - downscaled game screen, 3 frame stack
  - `health`: (1,) float32 - party HP fraction (0-1)
  - `level`: (8,) float32 - Fourier-encoded party level sum
  - `badges`: (8,) binary - badge bits
  - `events`: (311*8,) binary - event flag bits
  - `map`: (48, 48, 1) uint8 - local exploration map
  - `recent_actions`: (3,) discrete - last 3 actions

- **Action Space**: Discrete(7)
  - 0-3: Arrow keys (DOWN, LEFT, RIGHT, UP)
  - 4-5: A, B buttons
  - 6: START button

- **Episode Termination**:
  - Max steps reached (configurable via `max_steps`)
  - Task-specific: `badge_earned`, `pokecenter_reached` (optional)

---

## 2. Reward Shaping System

### Reward Components

The environment implements **4 separate reward channels** (defined in `env/reward_config.py`):

#### 2.1 Exploration Rewards
- **New tile**: +2.0 (configurable) when visiting a tile for the first time this episode
- **Recent tile**: +0.2 when revisiting a tile not in recent 50-step window
- **Tracked per episode**: `episode_visited_tiles` set is reset each episode

#### 2.2 Battle Rewards
- **HP delta**: ±2.0 × (opponent_hp_loss - player_hp_loss) during battle
- **Battle win**: +50.0 when opponent HP reaches 0
- **Battle loss**: -10.0 when player HP reaches 0
- **Battle detection**: Monitors RAM address `0xD057` (battle flag)

#### 2.3 Milestone Rewards
- **Badge earned**: +500.0 per badge
- **Level up**: +5.0 per level gained
- **Event flag**: +5.0 per new event flag set
- **Key location**: +30.0 when entering essential map (Pokecenter, Gym, etc.)

#### 2.4 Penalty Rewards
- **Step penalty**: -0.002 every step (encourages efficiency)
- **Wall collision**: -0.05 when position unchanged
- **Stuck penalty**: -0.1 when same tile visited >600 times

### Reward Aggregation

```python
step_reward = (
    exploration_rew +
    battle_rew +
    milestone_rew +
    penalty_rew
) * reward_scale
```

All components are **logged separately** to TensorBoard under `reward_components/`.

### Current Problem (User Report)

> "Agent mostly walks around, never battles, exploration dominates"

**Analysis**:
- Exploration rewards are **dense** (every new tile = +2.0)
- Battle rewards are **sparse** (only during infrequent battles)
- With `max_steps=2048`, agent can explore ~2048 tiles × 2.0 = **4096 total exploration reward**
- Battles require: finding grass/trainers, engaging, winning
- **Imbalance**: Walking is easier to learn than battling

---

## 3. PPO Training Script Flow

### Entry Point: `training/train_ppo.py`

#### 3.1 Configuration Loading
1. Parse CLI arguments (`argparse`)
2. Load JSON config from `configs/*.json`
3. Merge with defaults (`DEFAULT_ENV_CONFIG`, `DEFAULT_TRAIN_CONFIG`)
4. Validate configurations (`config_utils.py`)

#### 3.2 Environment Setup
```python
# Build N parallel environments
env = SubprocVecEnv([
    make_env(i, env_config, stream, seed)
    for i in range(num_envs)
])
env = VecMonitor(env)  # Add episode logging
```

#### 3.3 PPO Initialization
```python
model = PPO(
    "MultiInputPolicy",  # Dict observation space
    env,
    n_steps=rollout_horizon,  # Currently 512
    batch_size=batch_size,     # Config-dependent
    n_epochs=n_epochs,
    gamma=gamma,
    ent_coef=ent_coef,
    learning_rate=learning_rate,
    # ... more hyperparameters
    tensorboard_log=run_dir
)
```

**Current Settings** (from `train_ppo.py` defaults):
- `rollout_horizon`: 512 (reduced for memory)
- `total_multiplier`: 50 (default)
- `batch_size`: 512
- `n_epochs`: 1
- `gamma`: 0.997
- `ent_coef`: 0.01
- `learning_rate`: 3e-4

#### 3.4 Callbacks
- **CheckpointCallback**: Saves model every N steps
- **TensorboardCallback**: Custom logging (see below)
- **StatusWriterCallback**: Writes `status.json` for dashboard
- **PeriodicEvalCallback**: Optional eval during training

#### 3.5 Training Loop
```python
model.learn(
    total_timesteps=rollout_horizon * num_envs * total_multiplier,
    callback=CallbackList(callbacks)
)
```

---

## 4. Logging and Metrics

### TensorBoard Callback (`training/tensorboard_callback.py`)

#### 4.1 Episode-Level Metrics (logged at episode end)

**From VecMonitor** (`info['episode']`):
- `train/episode_return_mean`
- `train/episode_length_mean`
- `train/exploration_return`
- `train/battle_return`
- `train/milestone_return`
- `train/penalty_return`
- `train/success_rate` (if task defines success)

**From Environment** (`training_env.get_attr("agent_stats")`):
- `env_stats/step`, `env_stats/x`, `env_stats/y`, `env_stats/map`
- `env_stats/pcount` (party size)
- `env_stats/levels_sum`
- `env_stats/hp`
- `env_stats/badge`
- `env_stats/deaths`

**Reward Components** (`training_env.get_attr("episode_reward_components")`):
- `reward_components/exploration`
- `reward_components/battle`
- `reward_components/milestone`
- `reward_components/penalty`
- `reward_components/total_shaped`

**Battle Statistics** (`training_env.get_attr("episode_battle_stats")`):
- `battle_stats/wins_mean`
- `battle_stats/losses_mean`
- `battle_stats/total_mean`
- `battle_stats/win_rate_mean`

#### 4.2 Logging Trigger

Metrics are logged when **any environment** reaches `max_steps`:
```python
step_counts = self.training_env.get_attr("step_count")
max_steps = self.training_env.get_attr("max_steps")
if step_counts[0] >= max_steps[0] - 1:
    # Log all env_stats, reward_components, battle_stats
```

**Current Issue**: With `max_steps=2048`, episodes are short. Agent may not encounter battles.

---

## 5. Curriculum Tasks (Existing)

### Task Configurations (`configs/*.json`)

| Config | Objective | `max_steps` | Termination | Reward Focus |
|--------|-----------|-------------|-------------|--------------|
| `walk_to_pokecenter.json` | Reach Viridian PC | 40960 | Location | Exploration (10.0) |
| `exploration_basic.json` | Explore Pallet/Route 1 | 40960 | Max steps | Exploration (3.0) |
| `battle_training.json` | Win battles | 40960 | Max steps | Battle (5.0 HP, 10.0 win) |
| `gym_quest.json` | Earn first badge | **2048** | Badge | Balanced (1.0 scale) |
| `full_game_shaped.json` | Full game | 40960 | Max steps | All components |

### Config Structure
```json
{
  "env": {
    "headless": true,
    "max_steps": 2048,
    "reward_scale": 1.0,
    "termination_condition": "badge_earned",
    "reward_config": {
      "exploration_new_tile": 2.0,
      "battle_hp_delta": 2.0,
      "battle_win": 50.0,
      "milestone_badge": 500.0,
      // ... more reward coefficients
    }
  },
  "train": {
    "num_envs": 16,
    "total_multiplier": 100,
    "batch_size": 1024,
    "n_epochs": 4,
    "gamma": 0.995,
    // ... PPO hyperparameters
  }
}
```

---

## 6. Evaluation System

### Evaluation Script (`eval_policy.py`)

**Features**:
- Loads trained checkpoint (.zip)
- Runs N deterministic episodes
- Logs episode returns, lengths, success rate
- Prints battle statistics, badges, levels

**Usage**:
```bash
python eval_policy.py \
  --config configs/gym_quest.json \
  --checkpoint runs/my_run/poke_100000_steps.zip \
  --n_episodes 20
```

**Outputs**:
- Mean/std of episode return
- Mean/std of episode length
- Success rate (task-dependent)
- Battle win/loss counts

---

## 7. Key RAM Addresses (Pokemon Red)

The environment reads game state via PyBoy memory access:

| Address | Purpose | Used For |
|---------|---------|----------|
| `0xD362`, `0xD361`, `0xD35E` | Player X, Y, Map | Position tracking, exploration |
| `0xD057` | Battle flag | Battle detection |
| `0xD163` | Party size | Tracking team |
| `0xD18C`, `0xD1B8`, ... | Party levels | Milestone rewards |
| `0xD16C`, `0xD198`, ... | Party current HP | Health obs, battle rewards |
| `0xD18D`, `0xD1B9`, ... | Party max HP | HP fraction |
| `0xCFE6`, `0xCFF4` | Opponent HP | Battle reward (HP delta) |
| `0xD356` | Badge bits | Milestone rewards |
| `0xD747`-`0xD87E` | Event flags | Milestone rewards, state tracking |

---

## 8. Current Implementation Gaps (User-Reported Issues)

### Problem: "Agent doesn't battle"

**Root Causes**:
1. **Sparse battle encounters**: Battles require:
   - Walking through grass (random encounters)
   - Engaging trainers (specific map tiles)
   - Both are rare events in early game

2. **Reward imbalance**:
   - Exploration: dense, frequent (+2.0 per tile)
   - Battle: sparse, infrequent (+50.0 per win, but battles rare)
   - **Agent learns**: "walk around" > "find and win battles"

3. **Episode length**: `max_steps=2048` (gym_quest.json)
   - Too short to reach many battles
   - Agent hits max_steps before encountering meaningful events

4. **No battle-forcing mechanism**:
   - Environment doesn't guide agent toward grass/trainers
   - No curriculum that isolates battle skill

### Problem: "No meaningful progress"

**Root Causes**:
1. **Badge milestone too rare**:
   - First badge requires: walk to Viridian Forest, catch/train Pokemon, reach Pewter Gym, defeat Brock
   - This is a **long horizon task** (many thousands of steps)
   - Current episode length (2048) insufficient

2. **Exploration overshadows progress**:
   - Agent can get +2.0 × 2048 = **4096** just walking
   - Badges worth +500, but require complex chain of actions
   - **PPO struggles** to credit actions thousands of steps back

---

## 9. Proposed Improvements (Next Phases)

### Phase 2-3: Environment & Logging
- Add **battle-count metrics** per episode
- Track **steps-to-first-battle**
- Adjust `max_steps` to allow battles
- Create small-map variants (constrain to Pallet + Route 1)

### Phase 4: Reward Rebalancing
- **Reduce exploration** rewards (0.05 instead of 2.0)
- **Increase battle-start** rewards (+5.0 for entering battle)
- **Keep battle-win** high (+50.0)
- Add **idle penalty** for no events

### Phase 5: Curriculum
- **Battle-only env**: Isolated battle training (no navigation)
- **Nav-only env**: Walk to target (no battles)
- **Small-map env**: Pallet + Route 1, high encounter rate

### Phase 6-7: Training & Eval
- Tune PPO hyperparameters for sparse rewards
- Add automatic eval logging
- Checkpoint best models

---

## 10. File-Level Summary

### Core Environment
- `env/red_gym_env.py` (891 lines): Main Gym environment, reward computation, state tracking
- `env/reward_config.py` (140 lines): RewardConfig dataclass, coefficient management
- `env/global_map.py` (45 lines): Map coordinate conversion
- `env/stream_agent_wrapper.py` (75 lines): WebSocket streaming for visualization

### Training System
- `training/train_ppo.py` (453 lines): Main training script, PPO setup, argument parsing
- `training/tensorboard_callback.py` (183 lines): Custom TensorBoard logging
- `training/status_tracking.py`: Status JSON writer, periodic eval callback
- `training/config_utils.py`: Config validation

### Evaluation & Debug
- `eval_policy.py` (303 lines): Checkpoint evaluation script
- `debug_rewards.py` (260 lines): Reward shaping validator
- `tools/test_reward_shaping.py`: Automated reward testing
- `tools/compare_runs.py`: Checkpoint comparison
- `training/play_checkpoint.py`: Play policy with visualization

### Configuration
- `configs/walk_to_pokecenter.json`: Simple navigation task
- `configs/exploration_basic.json`: Exploration-focused task
- `configs/battle_training.json`: Battle-focused task
- `configs/gym_quest.json`: First badge task
- `configs/full_game_shaped.json`: Full game task

---

## END OF PHASE 1 DOCUMENTATION
