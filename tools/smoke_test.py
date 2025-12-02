import argparse
import random
from pathlib import Path

from env.red_gym_env import RedGymEnv


def parse_args():
    parser = argparse.ArgumentParser(description="Lightweight sanity check for ROM/state/env wiring.")
    parser.add_argument("--rom", type=Path, default=Path("PokemonRed.gb"), help="Path to Pokemon Red ROM.")
    parser.add_argument("--state", type=Path, default=Path("init.state"), help="Initial save state path.")
    parser.add_argument("--steps", type=int, default=128, help="Number of random steps to run.")
    parser.add_argument("--headless", action="store_true", default=True, help="Force headless mode.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if not args.rom.exists():
        raise FileNotFoundError(f"ROM not found at {args.rom}")
    if not args.state.exists():
        raise FileNotFoundError(f"State file not found at {args.state}")

    env_config = {
        "headless": args.headless,
        "save_final_state": False,
        "early_stop": False,
        "action_freq": 24,
        "init_state": str(args.state),
        "max_steps": args.steps + 1,
        "print_rewards": False,
        "save_video": False,
        "fast_video": True,
        "session_path": Path("session_smoke"),
        "gb_path": str(args.rom),
        "debug": False,
        "reward_scale": 0.5,
        "explore_weight": 0.25,
    }

    env = RedGymEnv(env_config)
    obs, info = env.reset()
    for step in range(args.steps):
        action = random.randrange(env.action_space.n)
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break

    env.close()
    print(f"Smoke test passed for {step+1} steps using ROM={args.rom}, state={args.state}")
