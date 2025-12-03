"""
Debug reward shaping by running random actions and printing reward breakdowns.

This script validates that all reward components are computed correctly.

Usage:
    python debug_rewards.py --task walk_to_pokecenter
    python debug_rewards.py --task walk_to_pokecenter --steps 1000
    python debug_rewards.py --task exploration_basic --episodes 3
"""

import argparse
import json
import sys
from pathlib import Path

# Add project root to path
REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

from env.red_gym_env import RedGymEnv
import numpy as np


def load_task_config(task_name: str, rom_path: Path, state_path: Path):
    """Load task configuration from JSON file."""
    config_path = REPO_ROOT / "configs" / f"{task_name}.json"

    if not config_path.exists():
        print(f"Error: Config file not found: {config_path}")
        print(f"Available configs:")
        for cfg in (REPO_ROOT / "configs").glob("*.json"):
            if cfg.name != "train_default.json":
                print(f"  - {cfg.stem}")
        sys.exit(1)

    with open(config_path, 'r') as f:
        task_config = json.load(f)

    # Extract env config
    env_config = task_config.get("env", {})

    # Add required paths
    env_config["gb_path"] = str(rom_path)
    env_config["init_state"] = str(state_path)
    env_config["session_path"] = REPO_ROOT / "debug_session"
    env_config["headless"] = True
    env_config["save_video"] = False
    env_config["print_rewards"] = False  # We'll print our own

    return env_config, task_config


def print_reward_breakdown(step: int, reward: float, env: RedGymEnv, episode_num: int = None):
    """Print detailed reward breakdown for current step."""
    components = env.episode_reward_components

    # Episode header if provided
    if episode_num is not None:
        print(f"\n{'='*80}")
        print(f"EPISODE {episode_num} - Step {step}")
        print(f"{'='*80}")
    else:
        print(f"\nStep {step}:")

    print(f"  Total Reward: {reward:8.4f}")
    print(f"  Components:")
    print(f"    Exploration:  {components['exploration']:8.4f}")
    print(f"    Battle:       {components['battle']:8.4f}")
    print(f"    Milestone:    {components['milestone']:8.4f}")
    print(f"    Penalty:      {components['penalty']:8.4f}")
    print(f"    Legacy:       {components['legacy']:8.4f}")

    # Game state
    x, y, map_id = env.get_game_coords()
    print(f"  Game State:")
    print(f"    Position:     ({x}, {y}, map={map_id})")
    print(f"    HP:           {env.read_hp_fraction():.2f}")
    print(f"    Badges:       {env.get_badges()}")
    print(f"    Tiles Seen:   {len(env.episode_visited_tiles)}")

    # Battle stats
    if components['battle'] != 0 or env.in_battle:
        print(f"  Battle Stats:")
        print(f"    In Battle:    {env.in_battle}")
        print(f"    Battles Won:  {env.episode_battle_stats['battles_won']}")
        print(f"    Battles Lost: {env.episode_battle_stats['battles_lost']}")


def debug_steps(task_name: str, rom_path: Path, state_path: Path, num_steps: int):
    """Debug rewards by running random actions for N steps."""
    print(f"="*80)
    print(f"DEBUGGING REWARDS: {task_name}")
    print(f"Running {num_steps} random steps")
    print(f"="*80)

    env_config, task_config = load_task_config(task_name, rom_path, state_path)

    print(f"\nTask Config:")
    print(f"  Max Steps:    {env_config.get('max_steps', 'N/A')}")
    print(f"  Termination:  {env_config.get('termination_condition', 'max_steps')}")

    if "reward_config" in env_config:
        print(f"\nReward Config:")
        rc = env_config["reward_config"]
        print(f"  Exploration:  new={rc.get('exploration_new_tile', 0)}, recent={rc.get('exploration_recent_tile', 0)}")
        print(f"  Battle:       hp_delta={rc.get('battle_hp_delta', 0)}, win={rc.get('battle_win', 0)}, loss={rc.get('battle_loss', 0)}")
        print(f"  Milestone:    badge={rc.get('milestone_badge', 0)}, level={rc.get('milestone_level_up', 0)}, location={rc.get('milestone_key_location', 0)}")
        print(f"  Penalty:      step={rc.get('penalty_step', 0)}, wall={rc.get('penalty_wall', 0)}, stuck={rc.get('penalty_stuck', 0)}")

    env = RedGymEnv(env_config)
    obs, info = env.reset()

    print(f"\nInitial State:")
    x, y, map_id = env.get_game_coords()
    print(f"  Position:     ({x}, {y}, map={map_id})")
    print(f"  HP:           {env.read_hp_fraction():.2f}")

    total_reward = 0
    for step in range(num_steps):
        action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(action)
        total_reward += reward

        # Print every 10 steps or if reward is non-zero
        if step % 10 == 0 or abs(reward) > 0.001:
            print_reward_breakdown(step, reward, env)

        if done or truncated:
            print(f"\n{'='*80}")
            print(f"Episode ended at step {step}")
            print(f"Reason: {'Done' if done else 'Truncated'}")
            print(f"Total Episode Reward: {total_reward:.4f}")
            print(f"{'='*80}")
            break

    # Final summary
    print(f"\n{'='*80}")
    print(f"FINAL SUMMARY")
    print(f"{'='*80}")
    print(f"Total Steps:         {step + 1}")
    print(f"Total Reward:        {total_reward:.4f}")
    print(f"Reward Components:")
    for comp, val in env.episode_reward_components.items():
        print(f"  {comp:12s}:    {val:8.4f}")
    print(f"Tiles Explored:      {len(env.episode_visited_tiles)}")
    print(f"Battles Won:         {env.episode_battle_stats['battles_won']}")
    print(f"Battles Lost:        {env.episode_battle_stats['battles_lost']}")


def debug_episodes(task_name: str, rom_path: Path, state_path: Path, num_episodes: int):
    """Debug rewards by running N complete episodes with random actions."""
    print(f"="*80)
    print(f"DEBUGGING REWARDS: {task_name}")
    print(f"Running {num_episodes} random episodes")
    print(f"="*80)

    env_config, task_config = load_task_config(task_name, rom_path, state_path)

    env = RedGymEnv(env_config)

    episode_returns = []
    episode_lengths = []

    for ep in range(num_episodes):
        obs, info = env.reset()
        episode_reward = 0
        step = 0

        print(f"\n{'='*80}")
        print(f"EPISODE {ep + 1}/{num_episodes}")
        print(f"{'='*80}")

        x, y, map_id = env.get_game_coords()
        print(f"Initial Position: ({x}, {y}, map={map_id})")

        while True:
            action = env.action_space.sample()
            obs, reward, done, truncated, info = env.step(action)
            episode_reward += reward
            step += 1

            # Print every 50 steps
            if step % 50 == 0:
                print(f"  Step {step}: reward={episode_reward:.2f}, tiles={len(env.episode_visited_tiles)}")

            if done or truncated:
                break

        episode_returns.append(episode_reward)
        episode_lengths.append(step)

        print(f"\nEpisode {ep + 1} Summary:")
        print(f"  Length:          {step}")
        print(f"  Total Reward:    {episode_reward:.4f}")
        print(f"  Reward Components:")
        for comp, val in env.episode_reward_components.items():
            if comp != 'legacy':
                print(f"    {comp:12s}: {val:8.4f}")
        print(f"  Tiles Explored:  {len(env.episode_visited_tiles)}")
        print(f"  Battles Won:     {env.episode_battle_stats['battles_won']}")
        print(f"  Battles Lost:    {env.episode_battle_stats['battles_lost']}")

    # Overall summary
    print(f"\n{'='*80}")
    print(f"OVERALL SUMMARY ({num_episodes} episodes)")
    print(f"{'='*80}")
    print(f"Mean Return:         {np.mean(episode_returns):.4f} ± {np.std(episode_returns):.4f}")
    print(f"Mean Length:         {np.mean(episode_lengths):.1f} ± {np.std(episode_lengths):.1f}")
    print(f"Max Return:          {np.max(episode_returns):.4f}")
    print(f"Min Return:          {np.min(episode_returns):.4f}")


def main():
    parser = argparse.ArgumentParser(description="Debug reward shaping with random actions")
    parser.add_argument("--task", type=str, default="walk_to_pokecenter",
                        help="Task name (config file name without .json)")
    parser.add_argument("--rom", type=Path, default=Path("PokemonRed.gb"),
                        help="Path to Pokemon Red ROM")
    parser.add_argument("--state", type=Path, default=Path("init.state"),
                        help="Path to initial state file")
    parser.add_argument("--steps", type=int, default=None,
                        help="Number of steps to run (mutually exclusive with --episodes)")
    parser.add_argument("--episodes", type=int, default=None,
                        help="Number of episodes to run (mutually exclusive with --steps)")

    args = parser.parse_args()

    # Validate paths
    if not args.rom.exists():
        print(f"Error: ROM not found at {args.rom}")
        sys.exit(1)
    if not args.state.exists():
        print(f"Error: State file not found at {args.state}")
        sys.exit(1)

    # Default: run 1 episode if neither specified
    if args.steps is None and args.episodes is None:
        args.episodes = 1

    # Run debugging
    if args.steps is not None:
        debug_steps(args.task, args.rom, args.state, args.steps)
    else:
        debug_episodes(args.task, args.rom, args.state, args.episodes)


if __name__ == "__main__":
    main()
