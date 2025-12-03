"""
Evaluate a trained PPO policy checkpoint.

This script loads a trained model and runs evaluation episodes to measure:
- Episode return (total reward)
- Episode length
- Success rate (task-specific)
- Optionally exports trajectories to JSON

Usage:
    python eval_policy.py --config configs/walk_to_pokecenter.json --checkpoint runs/my_run/poke_500000_steps.zip
    python eval_policy.py --config configs/walk_to_pokecenter.json --checkpoint runs/my_run/poke_500000_steps.zip --n_episodes 20
    python eval_policy.py --config configs/gym_quest.json --checkpoint runs/gym/final.zip --n_episodes 10 --export_trajectory
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import List, Dict, Any

# Add project root to path
REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

from env.red_gym_env import RedGymEnv
from stable_baselines3 import PPO
import numpy as np


def load_task_config(config_path: Path, rom_path: Path, state_path: Path):
    """Load task configuration from JSON file."""
    if not config_path.exists():
        print(f"Error: Config file not found: {config_path}")
        sys.exit(1)

    with open(config_path, 'r') as f:
        task_config = json.load(f)

    # Extract env config
    env_config = task_config.get("env", {})

    # Add required paths
    env_config["gb_path"] = str(rom_path)
    env_config["init_state"] = str(state_path)
    env_config["session_path"] = REPO_ROOT / "eval_session"
    env_config["headless"] = True
    env_config["save_video"] = False
    env_config["print_rewards"] = False

    return env_config, task_config


def check_success(env: RedGymEnv, done: bool, truncated: bool) -> bool:
    """
    Determine if episode was successful based on task termination condition.

    Args:
        env: The environment
        done: Whether episode is done
        truncated: Whether episode was truncated

    Returns:
        True if task objective was achieved
    """
    if not done and not truncated:
        return False

    # If episode ended due to success condition (not max steps)
    if done and not truncated:
        return True

    # Check task-specific success conditions
    if env.termination_condition:
        if env.termination_condition == 'badge_earned':
            # Success if badge was earned
            return env.get_badges() > 0
        elif env.termination_condition == 'pokecenter_reached':
            # Success if we're at Pokecenter map
            current_map = env.read_m(0xD35E)
            return current_map == 40

    # Default: not successful if truncated
    return done and not truncated


def run_evaluation(
    model: PPO,
    env: RedGymEnv,
    n_episodes: int,
    max_steps_per_episode: int,
    export_trajectory: bool = False,
    render: bool = False
) -> Dict[str, Any]:
    """
    Run evaluation episodes with the trained policy.

    Args:
        model: Trained PPO model
        env: Environment to evaluate on
        n_episodes: Number of episodes to run
        max_steps_per_episode: Maximum steps per episode
        export_trajectory: Whether to export detailed trajectories
        render: Whether to render (not implemented for headless)

    Returns:
        Dictionary with evaluation statistics
    """
    episode_returns = []
    episode_lengths = []
    successes = []
    trajectories = []

    print(f"Running {n_episodes} evaluation episodes...")
    print(f"Max steps per episode: {max_steps_per_episode}")

    for ep in range(n_episodes):
        obs, info = env.reset()
        episode_reward = 0
        episode_steps = 0
        trajectory = [] if export_trajectory else None

        print(f"\nEpisode {ep + 1}/{n_episodes}:")
        x, y, map_id = env.get_game_coords()
        print(f"  Start position: ({x}, {y}, map={map_id})")

        while episode_steps < max_steps_per_episode:
            # Get action from policy
            action, _states = model.predict(obs, deterministic=True)

            # Store trajectory if requested
            if export_trajectory:
                x, y, map_id = env.get_game_coords()
                trajectory.append({
                    'step': episode_steps,
                    'position': (int(x), int(y), int(map_id)),
                    'action': int(action),
                    'hp': float(env.read_hp_fraction()),
                    'badges': int(env.get_badges()),
                })

            # Take action
            obs, reward, done, truncated, info = env.step(action)
            episode_reward += reward
            episode_steps += 1

            # Progress indicator
            if episode_steps % 100 == 0:
                tiles = len(env.episode_visited_tiles)
                print(f"    Step {episode_steps}: reward={episode_reward:.2f}, tiles={tiles}")

            if done or truncated:
                break

        # Record episode statistics
        episode_returns.append(episode_reward)
        episode_lengths.append(episode_steps)
        success = check_success(env, done, truncated)
        successes.append(success)

        # Add final trajectory entry
        if export_trajectory:
            x, y, map_id = env.get_game_coords()
            trajectory.append({
                'step': episode_steps,
                'position': (int(x), int(y), int(map_id)),
                'action': None,
                'hp': float(env.read_hp_fraction()),
                'badges': int(env.get_badges()),
                'success': success,
                'total_reward': float(episode_reward),
            })
            trajectories.append(trajectory)

        # Print episode summary
        print(f"  End position:   ({x}, {y}, map={map_id})")
        print(f"  Episode length: {episode_steps}")
        print(f"  Episode return: {episode_reward:.4f}")
        print(f"  Success:        {'✓' if success else '✗'}")
        print(f"  Reward breakdown:")
        for comp, val in env.episode_reward_components.items():
            if comp != 'legacy':
                print(f"    {comp:12s}: {val:8.4f}")

    # Compute statistics
    results = {
        'n_episodes': n_episodes,
        'mean_return': float(np.mean(episode_returns)),
        'std_return': float(np.std(episode_returns)),
        'mean_length': float(np.mean(episode_lengths)),
        'std_length': float(np.std(episode_lengths)),
        'success_rate': float(np.mean(successes)),
        'max_return': float(np.max(episode_returns)),
        'min_return': float(np.min(episode_returns)),
        'episode_returns': [float(r) for r in episode_returns],
        'episode_lengths': [int(l) for l in episode_lengths],
        'successes': [bool(s) for s in successes],
    }

    if export_trajectory:
        results['trajectories'] = trajectories

    return results


def print_results(results: Dict[str, Any]):
    """Print evaluation results in a readable format."""
    print(f"\n{'='*80}")
    print(f"EVALUATION RESULTS")
    print(f"{'='*80}")
    print(f"Episodes:            {results['n_episodes']}")
    print(f"Mean Return:         {results['mean_return']:8.4f} ± {results['std_return']:.4f}")
    print(f"Mean Length:         {results['mean_length']:8.1f} ± {results['std_length']:.1f}")
    print(f"Success Rate:        {results['success_rate']*100:6.2f}%")
    print(f"Max Return:          {results['max_return']:8.4f}")
    print(f"Min Return:          {results['min_return']:8.4f}")
    print(f"\nPer-Episode Results:")
    for i, (ret, length, success) in enumerate(zip(
        results['episode_returns'],
        results['episode_lengths'],
        results['successes']
    )):
        status = '✓' if success else '✗'
        print(f"  Episode {i+1:2d}: return={ret:8.4f}, length={length:4d}, success={status}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained PPO policy")
    parser.add_argument("--config", type=Path, required=True,
                        help="Path to task config JSON")
    parser.add_argument("--checkpoint", type=Path, required=True,
                        help="Path to trained model checkpoint (.zip)")
    parser.add_argument("--rom", type=Path, default=Path("PokemonRed.gb"),
                        help="Path to Pokemon Red ROM")
    parser.add_argument("--state", type=Path, default=Path("init.state"),
                        help="Path to initial state file")
    parser.add_argument("--n_episodes", type=int, default=10,
                        help="Number of evaluation episodes")
    parser.add_argument("--max_steps", type=int, default=None,
                        help="Max steps per episode (default: from config)")
    parser.add_argument("--export_trajectory", action="store_true",
                        help="Export detailed trajectories to JSON")
    parser.add_argument("--output", type=Path, default=None,
                        help="Output JSON file for results (default: eval_results_<timestamp>.json)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for evaluation")

    args = parser.parse_args()

    # Validate paths
    if not args.rom.exists():
        print(f"Error: ROM not found at {args.rom}")
        sys.exit(1)
    if not args.state.exists():
        print(f"Error: State file not found at {args.state}")
        sys.exit(1)
    if not args.checkpoint.exists():
        print(f"Error: Checkpoint not found at {args.checkpoint}")
        sys.exit(1)
    if not args.config.exists():
        print(f"Error: Config not found at {args.config}")
        sys.exit(1)

    # Set seed
    np.random.seed(args.seed)

    # Load configuration
    print(f"Loading task config: {args.config}")
    env_config, task_config = load_task_config(args.config, args.rom, args.state)

    # Get max steps from config if not specified
    max_steps = args.max_steps or env_config.get('max_steps', 10000)

    # Create environment
    print(f"Creating environment...")
    env = RedGymEnv(env_config)

    # Load model
    print(f"Loading checkpoint: {args.checkpoint}")
    try:
        model = PPO.load(str(args.checkpoint), env=env)
        print(f"Model loaded successfully")
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)

    # Run evaluation
    start_time = time.time()
    results = run_evaluation(
        model=model,
        env=env,
        n_episodes=args.n_episodes,
        max_steps_per_episode=max_steps,
        export_trajectory=args.export_trajectory,
    )
    elapsed_time = time.time() - start_time

    # Print results
    print_results(results)
    print(f"\nEvaluation completed in {elapsed_time:.1f} seconds")

    # Save results
    if args.output:
        output_path = args.output
    else:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_path = REPO_ROOT / f"eval_results_{timestamp}.json"

    results['config'] = args.config.stem
    results['checkpoint'] = str(args.checkpoint)
    results['timestamp'] = time.strftime("%Y-%m-%d %H:%M:%S")
    results['elapsed_time'] = elapsed_time

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
