import argparse
import time
import sys
from pathlib import Path
from typing import Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from env.red_gym_env import RedGymEnv
from env.stream_agent_wrapper import StreamWrapper
from stable_baselines3 import PPO


def find_latest_checkpoint(run_dir: Path) -> Optional[Tuple[Path, float]]:
    zip_files = list(run_dir.glob("**/*.zip"))
    if not zip_files:
        return None
    most_recent = max(zip_files, key=lambda p: p.stat().st_mtime)
    age_hours = (time.time() - most_recent.stat().st_mtime) / 3600
    return most_recent, age_hours


def parse_args():
    parser = argparse.ArgumentParser(description="Play a trained checkpoint interactively.")
    parser.add_argument("--checkpoint", type=Path, default=None, help="Path to checkpoint .zip. Defaults to latest in runs/.")
    parser.add_argument("--runs-dir", type=Path, default=Path("runs"), help="Directory to search for checkpoints.")
    parser.add_argument("--rom", type=Path, default=Path("PokemonRed.gb"), help="Path to Pokemon Red ROM.")
    parser.add_argument("--state", type=Path, default=Path("init.state"), help="Initial save state path.")
    parser.add_argument("--headless", action="store_true", help="Run without SDL window.")
    parser.add_argument("--stream", action="store_true", default=True, help="Enable map streaming.")
    parser.add_argument("--no-stream", dest="stream", action="store_false", help="Disable map streaming.")
    parser.add_argument("--steps", type=int, default=None, help="Stop after this many steps (default: full episode).")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if not args.rom.exists():
        raise FileNotFoundError(f"ROM not found at {args.rom}")
    if not args.state.exists():
        raise FileNotFoundError(f"State file not found at {args.state}")

    checkpoint = args.checkpoint
    if checkpoint is None:
        latest = find_latest_checkpoint(args.runs_dir)
        if latest is None:
            raise FileNotFoundError("No checkpoint found. Provide --checkpoint or place .zip files under runs/.")
        checkpoint, age = latest
        print(f"Using latest checkpoint: {checkpoint} (age: {age:.2f} hours)")
    else:
        if not checkpoint.exists():
            raise FileNotFoundError(f"Checkpoint not found at {checkpoint}")

    ep_length = 2**23
    env_config = {
        "headless": args.headless,
        "save_final_state": True,
        "early_stop": False,
        "action_freq": 24,
        "init_state": str(args.state),
        "max_steps": ep_length,
        "print_rewards": True,
        "save_video": False,
        "fast_video": True,
        "session_path": Path("session_interactive"),
        "gb_path": str(args.rom),
        "debug": False,
        "reward_scale": 0.5,
        "explore_weight": 0.25,
    }

    base_env = RedGymEnv(env_config)
    if args.stream:
        env = StreamWrapper(
            base_env,
            stream_metadata={
                "user": "interactive",
                "env_id": 0,
                "color": "#33aa33",
                "extra": "",
            },
        )
    else:
        env = base_env

    model = PPO.load(str(checkpoint), env=env, custom_objects={"lr_schedule": 0, "clip_range": 0})

    obs, info = env.reset()
    max_steps = args.steps or ep_length
    steps = 0
    while True:
        action, _states = model.predict(obs, deterministic=False)
        obs, rewards, terminated, truncated, info = env.step(action)
        env.render()
        steps += 1
        if terminated or truncated or steps >= max_steps:
            break
    env.close()
