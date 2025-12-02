import argparse
import json
import time
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
from env.red_gym_env import RedGymEnv
from stable_baselines3 import PPO


def parse_args():
    parser = argparse.ArgumentParser(description="Compare two checkpoints side-by-side with quick rollouts.")
    parser.add_argument("--checkpoint-a", type=Path, required=True, help="Path to first checkpoint .zip")
    parser.add_argument("--checkpoint-b", type=Path, required=True, help="Path to second checkpoint .zip")
    parser.add_argument("--rom", type=Path, default=Path("PokemonRed.gb"), help="Path to Pokemon Red ROM.")
    parser.add_argument("--state", type=Path, default=Path("init.state"), help="Initial save state path.")
    parser.add_argument("--steps", type=int, default=512, help="Steps to roll each checkpoint for comparison.")
    parser.add_argument("--output", type=Path, default=None, help="Directory to write comparison outputs.")
    return parser.parse_args()


def run_eval(checkpoint: Path, rom: Path, state: Path, steps: int, out_dir: Path) -> Tuple[Dict, np.ndarray, np.ndarray]:
    env_config = {
        "headless": True,
        "save_final_state": False,
        "early_stop": False,
        "action_freq": 24,
        "init_state": str(state),
        "max_steps": steps + 1,
        "print_rewards": False,
        "save_video": False,
        "fast_video": True,
        "session_path": out_dir / f"session_{checkpoint.stem}",
        "gb_path": str(rom),
        "debug": False,
        "reward_scale": 0.5,
        "explore_weight": 0.25,
    }

    env = RedGymEnv(env_config)
    model = PPO.load(str(checkpoint), env=env, custom_objects={"lr_schedule": 0, "clip_range": 0})

    obs, info = env.reset()
    total_reward = 0.0
    for step in range(steps):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += float(reward)
        if terminated or truncated:
            break

    screen = env.render()  # latest screen
    explore_map = env.get_explore_map()
    coverage = int(np.count_nonzero(explore_map))

    stats = {
        "checkpoint": str(checkpoint),
        "steps_run": step + 1,
        "total_reward": total_reward,
        "badges": env.get_badges(),
        "coverage_pixels": coverage,
        "max_map_progress": env.max_map_progress,
    }

    env.close()
    return stats, screen, explore_map


def plot_comparison(screen_a, map_a, screen_b, map_b, out_path: Path):
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes[0, 0].imshow(screen_a.squeeze(), cmap="gray")
    axes[0, 0].set_title("Checkpoint A screen")
    axes[0, 1].imshow(map_a, cmap="inferno")
    axes[0, 1].set_title("Checkpoint A map")
    axes[1, 0].imshow(screen_b.squeeze(), cmap="gray")
    axes[1, 0].set_title("Checkpoint B screen")
    axes[1, 1].imshow(map_b, cmap="inferno")
    axes[1, 1].set_title("Checkpoint B map")
    for ax in axes.ravel():
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


if __name__ == "__main__":
    args = parse_args()

    for p in [args.checkpoint_a, args.checkpoint_b]:
        if not p.exists():
            raise FileNotFoundError(f"Checkpoint not found: {p}")
    if not args.rom.exists():
        raise FileNotFoundError(f"ROM not found at {args.rom}")
    if not args.state.exists():
        raise FileNotFoundError(f"State file not found at {args.state}")

    out_dir = args.output or Path("runs") / f"compare_{int(time.time())}"
    out_dir.mkdir(parents=True, exist_ok=True)

    stats_a, screen_a, map_a = run_eval(args.checkpoint_a, args.rom, args.state, args.steps, out_dir)
    stats_b, screen_b, map_b = run_eval(args.checkpoint_b, args.rom, args.state, args.steps, out_dir)

    comp_png = out_dir / "comparison.png"
    plot_comparison(screen_a, map_a.squeeze(), screen_b, map_b.squeeze(), comp_png)

    summary = {"run_a": stats_a, "run_b": stats_b, "comparison_image": str(comp_png)}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    print(f"Wrote comparison to {out_dir}")
