"""
Test script to validate reward shaping implementation.

This script:
1. Creates test environments with different reward configs
2. Runs random actions for a few steps
3. Prints reward component breakdowns
4. Verifies that rewards behave as expected
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from env.red_gym_env import RedGymEnv
from env.reward_config import RewardConfig, get_reward_config
import numpy as np


def test_exploration_rewards(rom_path, state_path):
    """Test that exploration rewards work correctly."""
    print("\n" + "=" * 60)
    print("Testing Exploration Rewards")
    print("=" * 60)

    config = {
        "session_path": Path("./test_session"),
        "gb_path": str(rom_path),
        "init_state": str(state_path),
        "headless": True,
        "save_final_state": False,
        "action_freq": 24,
        "max_steps": 100,
        "print_rewards": False,
        "save_video": False,
        "fast_video": True,
        "reward_config": get_reward_config('exploration'),
    }

    env = RedGymEnv(config)
    obs, info = env.reset()

    print(f"Initial position: {env.get_game_coords()}")
    print(f"Episode visited tiles: {len(env.episode_visited_tiles)}")

    # Take some random actions
    total_reward = 0
    for step in range(20):
        action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(action)
        total_reward += reward

        if step % 5 == 0:
            print(f"\nStep {step}:")
            print(f"  Position: {env.get_game_coords()}")
            print(f"  Episode tiles visited: {len(env.episode_visited_tiles)}")
            print(f"  Step reward: {reward:.3f}")
            print(f"  Reward components: exploration={env.episode_reward_components['exploration']:.2f}, "
                  f"battle={env.episode_reward_components['battle']:.2f}, "
                  f"milestone={env.episode_reward_components['milestone']:.2f}, "
                  f"penalty={env.episode_reward_components['penalty']:.2f}")

        if done or truncated:
            break

    print(f"\nFinal episode reward: {total_reward:.3f}")
    print(f"Final component breakdown:")
    for comp, val in env.episode_reward_components.items():
        if comp != 'legacy':
            print(f"  {comp}: {val:.3f}")

    # Verify exploration rewards are positive
    assert env.episode_reward_components['exploration'] > 0, "Exploration rewards should be positive!"
    print("\n✓ Exploration rewards working correctly!")


def test_battle_rewards(rom_path, state_path):
    """Test that battle rewards work correctly (basic check)."""
    print("\n" + "=" * 60)
    print("Testing Battle Rewards Setup")
    print("=" * 60)

    config = {
        "session_path": Path("./test_session"),
        "gb_path": str(rom_path),
        "init_state": str(state_path),
        "headless": True,
        "save_final_state": False,
        "action_freq": 24,
        "max_steps": 100,
        "print_rewards": False,
        "save_video": False,
        "fast_video": True,
        "reward_config": get_reward_config('battle'),
    }

    env = RedGymEnv(config)
    obs, info = env.reset()

    print(f"Battle config: hp_delta={env.reward_config.battle_hp_delta}, "
          f"win={env.reward_config.battle_win}, loss={env.reward_config.battle_loss}")
    print(f"In battle: {env.in_battle}")
    print(f"Player HP: {env.read_hp_fraction():.2f}")

    # Take a few random actions
    for step in range(10):
        action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(action)

        if step % 3 == 0:
            print(f"\nStep {step}: in_battle={env.in_battle}, player_hp={env.read_hp_fraction():.2f}")

        if done or truncated:
            break

    print(f"\nBattle component: {env.episode_reward_components['battle']:.3f}")
    print("✓ Battle reward system initialized correctly!")


def test_milestone_rewards(rom_path, state_path):
    """Test that milestone rewards are configured correctly."""
    print("\n" + "=" * 60)
    print("Testing Milestone Rewards Setup")
    print("=" * 60)

    config = {
        "session_path": Path("./test_session"),
        "gb_path": str(rom_path),
        "init_state": str(state_path),
        "headless": True,
        "save_final_state": False,
        "action_freq": 24,
        "max_steps": 100,
        "print_rewards": False,
        "save_video": False,
        "fast_video": True,
        "reward_config": get_reward_config('milestone'),
    }

    env = RedGymEnv(config)
    obs, info = env.reset()

    print(f"Milestone config:")
    print(f"  Badge: {env.reward_config.milestone_badge}")
    print(f"  Level up: {env.reward_config.milestone_level_up}")
    print(f"  Key location: {env.reward_config.milestone_key_location}")
    print(f"  Event: {env.reward_config.milestone_event}")

    print(f"\nCurrent state:")
    print(f"  Badges: {env.get_badges()}")
    print(f"  Party levels: {env.get_party_levels()}")
    print(f"  Events: {env.get_all_events_reward()}")

    print("✓ Milestone reward system initialized correctly!")


def test_penalty_rewards(rom_path, state_path):
    """Test that penalties are applied correctly."""
    print("\n" + "=" * 60)
    print("Testing Penalty Rewards")
    print("=" * 60)

    config = {
        "session_path": Path("./test_session"),
        "gb_path": str(rom_path),
        "init_state": str(state_path),
        "headless": True,
        "save_final_state": False,
        "action_freq": 24,
        "max_steps": 100,
        "print_rewards": False,
        "save_video": False,
        "fast_video": True,
        "reward_config": RewardConfig(
            exploration_new_tile=0.0,  # Disable other rewards
            battle_hp_delta=0.0,
            milestone_badge=0.0,
            penalty_step=-0.01,
            penalty_wall=-0.1,
            penalty_stuck=-0.05,
        ),
    }

    env = RedGymEnv(config)
    obs, info = env.reset()

    print(f"Penalty config: step={env.reward_config.penalty_step}, "
          f"wall={env.reward_config.penalty_wall}, stuck={env.reward_config.penalty_stuck}")

    # Take some actions
    total_penalty = 0
    for step in range(20):
        action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(action)
        total_penalty += reward

        if done or truncated:
            break

    print(f"\nTotal penalty after 20 steps: {total_penalty:.3f}")
    print(f"Penalty component: {env.episode_reward_components['penalty']:.3f}")

    # Verify penalties are negative
    assert env.episode_reward_components['penalty'] < 0, "Penalties should be negative!"
    print("✓ Penalty rewards working correctly!")


def test_all_configs(rom_path, state_path):
    """Test all predefined reward configurations."""
    print("\n" + "=" * 60)
    print("Testing All Reward Configurations")
    print("=" * 60)

    configs = ['default', 'exploration', 'battle', 'milestone', 'minimal_penalty']

    for config_name in configs:
        print(f"\n{config_name}:")
        reward_config = get_reward_config(config_name)
        print(f"  Exploration: new={reward_config.exploration_new_tile}, recent={reward_config.exploration_recent_tile}")
        print(f"  Battle: hp_delta={reward_config.battle_hp_delta}, win={reward_config.battle_win}")
        print(f"  Milestone: badge={reward_config.milestone_badge}, level={reward_config.milestone_level_up}")
        print(f"  Penalty: step={reward_config.penalty_step}, wall={reward_config.penalty_wall}")

    print("\n✓ All reward configurations loaded successfully!")


def main():
    parser = argparse.ArgumentParser(description="Test reward shaping implementation")
    parser.add_argument("--rom", type=Path, default=Path("PokemonRed.gb"), help="Path to ROM")
    parser.add_argument("--state", type=Path, default=Path("init.state"), help="Path to initial state")
    args = parser.parse_args()

    if not args.rom.exists():
        print(f"Error: ROM not found at {args.rom}")
        sys.exit(1)
    if not args.state.exists():
        print(f"Error: State file not found at {args.state}")
        sys.exit(1)

    print("=" * 60)
    print("Reward Shaping Validation Tests")
    print("=" * 60)

    try:
        test_all_configs(args.rom, args.state)
        test_exploration_rewards(args.rom, args.state)
        test_battle_rewards(args.rom, args.state)
        test_milestone_rewards(args.rom, args.state)
        test_penalty_rewards(args.rom, args.state)

        print("\n" + "=" * 60)
        print("✓ All tests passed!")
        print("=" * 60)

    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
