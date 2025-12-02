# Train RL agents to play Pokemon Red (Windows + NVIDIA)

Single V2 codepath using PyBoy + Stable Baselines3 PPO, streaming progress to the shared map by default.

## Quick start
- Requirements: Python 3.10+, ffmpeg on PATH, Windows with NVIDIA GPU/CUDA.
- Drop your legally obtained `PokemonRed.gb` in the repo root (sha1: `ea9bcae617fdf159b045185467ae58b2e4a48b9a`). `init.state` is already included for fast starts.
- Install deps: `pip install -r requirements.txt`

### Train (streams map by default)
```bash
python training/train_ppo.py --rom PokemonRed.gb --state init.state --run-name my_run
```
Config defaults live in `configs/train_default.json`; override with CLI flags (e.g., `--num-envs`, `--total-multiplier`, `--no-stream`).
Use GPU presets with `--preset small|medium|large` or enable wandb with `--wandb`.

### Play a checkpoint
```bash
python training/play_checkpoint.py --checkpoint runs/my_run/poke_XXXXX_steps.zip --rom PokemonRed.gb --state init.state
```
If `--checkpoint` is omitted, the script loads the most recent `.zip` under `runs/`. Add `--headless` to hide the SDL window; `--no-stream` to stay offline.

### Logs & metrics
- Checkpoints and TensorBoard logs: `runs/<run_name>/`
- View metrics: `tensorboard --logdir runs/<run_name>`

### Quick sanity check
```bash
python tools/smoke_test.py --rom PokemonRed.gb --state init.state
```

### Compare two checkpoints
```bash
python tools/compare_runs.py --checkpoint-a runs/runA/poke_XXX_steps.zip --checkpoint-b runs/runB/poke_YYY_steps.zip --rom PokemonRed.gb --state init.state
```
Writes a side-by-side PNG and summary JSON in `runs/compare_<timestamp>/`.

### Simple dashboard (no DS background needed)
```bash
python tools/serve_dashboard.py --port 8000
```
Opens a lightweight HTML dashboard listing runs (latest steps, streaming on/off, batch/num_envs) and shows a copy-paste compare command for any two runs.

## Layout
- `env/`: Gym environment (`red_gym_env.py`), map streamer (`stream_agent_wrapper.py`), map data (`map_data.json`, `events.json`, `global_map.py`)
- `training/`: PPO train/play scripts and `tensorboard_callback.py`
- `configs/`: `train_default.json` with env/train defaults
- `assets/`: README imagery

## Watch the demo
<p float="left">
  <a href="https://youtu.be/DcYLT37ImBY">
    <img src="/assets/youtube.jpg?raw=true" height="192">
  </a>
  <a href="https://youtu.be/DcYLT37ImBY">
    <img src="/assets/poke_map.gif?raw=true" height="192">
  </a>
</p>
