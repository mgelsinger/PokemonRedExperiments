"""
Curriculum task system for Pokemon Red RL training.

Defines different training tasks/scenarios with specific:
- Starting states
- Termination conditions
- Reward configurations
- Episode length limits

This allows training on focused sub-tasks before tackling the full game.
"""

from dataclasses import dataclass
from typing import Optional, Callable
from pathlib import Path
from .reward_config import RewardConfig, get_reward_config


@dataclass
class TaskConfig:
    """Configuration for a specific training task."""

    name: str
    description: str

    # Reward configuration for this task
    reward_config: RewardConfig

    # Episode constraints
    max_steps: int = 163840  # Default from train_default.json

    # Starting state (optional - use init.state if None)
    start_state_path: Optional[Path] = None

    # Termination condition (optional - returns True if episode should end)
    # This is a string identifier for the termination condition
    termination_condition: Optional[str] = None

    # Success condition (optional - for evaluation metrics)
    success_condition: Optional[str] = None

    def to_dict(self):
        """Convert to dictionary for JSON serialization."""
        return {
            'name': self.name,
            'description': self.description,
            'reward_config': self.reward_config.to_dict(),
            'max_steps': self.max_steps,
            'start_state_path': str(self.start_state_path) if self.start_state_path else None,
            'termination_condition': self.termination_condition,
            'success_condition': self.success_condition,
        }


# === Task Definitions ===

def get_exploration_task() -> TaskConfig:
    """
    Task: Explore Pallet Town and surrounding areas.
    Goal: Visit as many tiles as possible.
    """
    return TaskConfig(
        name='exploration_basic',
        description='Explore Pallet Town and Route 1. Focus on discovering new tiles.',
        reward_config=RewardConfig(
            # Heavy exploration focus
            exploration_new_tile=5.0,
            exploration_recent_tile=0.5,
            # Minimal battle rewards
            battle_hp_delta=0.0,
            battle_win=0.0,
            battle_loss=0.0,
            # Milestone rewards
            milestone_badge=50.0,
            milestone_level_up=1.0,
            milestone_key_location=20.0,
            milestone_event=2.0,
            # Penalties
            penalty_step=-0.001,
            penalty_wall=-0.05,
            penalty_stuck=-0.1,
            reward_scale=1.0,
        ),
        max_steps=2048 * 40,  # Shorter episodes for focused exploration
    )


def get_battle_training_task() -> TaskConfig:
    """
    Task: Train battle skills.
    Goal: Win battles efficiently without losing too much HP.
    """
    return TaskConfig(
        name='battle_training',
        description='Train battle skills. Win battles while minimizing HP loss.',
        reward_config=RewardConfig(
            # Minimal exploration
            exploration_new_tile=0.5,
            exploration_recent_tile=0.0,
            # Heavy battle focus
            battle_hp_delta=5.0,
            battle_win=100.0,
            battle_loss=-20.0,
            # Milestone rewards
            milestone_badge=100.0,
            milestone_level_up=10.0,
            milestone_key_location=5.0,
            milestone_event=2.0,
            # Penalties
            penalty_step=-0.005,
            penalty_wall=-0.05,
            penalty_stuck=-0.2,
            reward_scale=1.0,
        ),
        max_steps=2048 * 60,
    )


def get_gym_quest_task() -> TaskConfig:
    """
    Task: Reach and defeat the first gym.
    Goal: Navigate to Pewter City Gym and earn the Boulder Badge.
    """
    return TaskConfig(
        name='gym_quest',
        description='Navigate to Pewter City Gym and earn the Boulder Badge.',
        reward_config=RewardConfig(
            # Balanced exploration and combat
            exploration_new_tile=2.0,
            exploration_recent_tile=0.2,
            # Strong battle rewards
            battle_hp_delta=2.0,
            battle_win=50.0,
            battle_loss=-10.0,
            # Heavy milestone focus
            milestone_badge=500.0,  # Huge reward for badge
            milestone_level_up=5.0,
            milestone_key_location=30.0,
            milestone_event=5.0,
            # Penalties
            penalty_step=-0.002,
            penalty_wall=-0.05,
            penalty_stuck=-0.1,
            reward_scale=1.0,
        ),
        max_steps=2048 * 120,
        termination_condition='badge_earned',
        success_condition='badge_earned',
    )


def get_early_game_task() -> TaskConfig:
    """
    Task: Complete early game objectives.
    Goal: Get starter, explore Pallet/Viridian/Pewter, earn first badge.
    """
    return TaskConfig(
        name='early_game',
        description='Complete early game: Pallet Town -> Viridian -> Pewter City -> Boulder Badge.',
        reward_config=RewardConfig(
            # Balanced rewards
            exploration_new_tile=2.0,
            exploration_recent_tile=0.3,
            battle_hp_delta=1.0,
            battle_win=30.0,
            battle_loss=-5.0,
            milestone_badge=300.0,
            milestone_level_up=3.0,
            milestone_key_location=20.0,
            milestone_event=5.0,
            penalty_step=-0.002,
            penalty_wall=-0.05,
            penalty_stuck=-0.1,
            reward_scale=1.0,
        ),
        max_steps=2048 * 150,
    )


def get_full_game_task() -> TaskConfig:
    """
    Task: Play the full game.
    Goal: Explore all of Kanto, earn badges, defeat Elite Four.
    """
    return TaskConfig(
        name='full_game',
        description='Full game playthrough. Explore Kanto, earn all badges, defeat Elite Four.',
        reward_config=RewardConfig(
            # Balanced rewards for long-term play
            exploration_new_tile=1.0,
            exploration_recent_tile=0.1,
            battle_hp_delta=0.5,
            battle_win=20.0,
            battle_loss=-5.0,
            milestone_badge=200.0,
            milestone_level_up=2.0,
            milestone_key_location=10.0,
            milestone_event=4.0,
            penalty_step=-0.001,
            penalty_wall=-0.05,
            penalty_stuck=-0.05,
            reward_scale=0.5,  # Scale down for longer episodes
        ),
        max_steps=2048 * 200,
    )


def get_full_game_shaped_task() -> TaskConfig:
    """
    Task: Full game with strong reward shaping.
    Goal: Same as full game but with more aggressive reward shaping.
    """
    return TaskConfig(
        name='full_game_shaped',
        description='Full game with aggressive reward shaping for faster learning.',
        reward_config=RewardConfig(
            exploration_new_tile=2.0,
            exploration_recent_tile=0.2,
            battle_hp_delta=1.0,
            battle_win=50.0,
            battle_loss=-10.0,
            milestone_badge=300.0,
            milestone_level_up=5.0,
            milestone_key_location=20.0,
            milestone_event=8.0,
            penalty_step=-0.005,
            penalty_wall=-0.1,
            penalty_stuck=-0.2,
            reward_scale=1.0,
        ),
        max_steps=2048 * 200,
    )


# Task registry
TASKS = {
    'exploration_basic': get_exploration_task,
    'battle_training': get_battle_training_task,
    'gym_quest': get_gym_quest_task,
    'early_game': get_early_game_task,
    'full_game': get_full_game_task,
    'full_game_shaped': get_full_game_shaped_task,
}


def get_task(name: str) -> TaskConfig:
    """
    Get a task configuration by name.

    Args:
        name: Name of the task (e.g., 'exploration_basic', 'gym_quest', 'full_game')

    Returns:
        TaskConfig instance

    Raises:
        ValueError: If task name is not found
    """
    if name not in TASKS:
        raise ValueError(f"Unknown task: {name}. Available: {list(TASKS.keys())}")
    return TASKS[name]()


def list_tasks():
    """List all available tasks."""
    return list(TASKS.keys())
