"""
Reward shaping configuration for Pokemon Red RL training.

This module defines the reward structure with clear component breakdown:
- Exploration rewards: encourage discovering new areas
- Battle rewards: reward effective combat (HP management, wins)
- Milestone rewards: reward key achievements (badges, levels, items)
- Penalty rewards: discourage inefficient behavior (step penalty, wall collisions)
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, Any
import json
from pathlib import Path


@dataclass
class RewardConfig:
    """Configuration for reward shaping components."""

    # === Exploration Rewards ===
    # Coefficient for visiting new tiles (not seen this episode)
    exploration_new_tile: float = 1.0
    # Coefficient for visiting tiles not seen recently (last N steps)
    exploration_recent_tile: float = 0.1
    # Window size for "recent" tiles (in steps)
    exploration_recent_window: int = 100

    # === Battle Rewards ===
    # Coefficient for HP delta (opponent_hp_loss - player_hp_loss)
    battle_hp_delta: float = 0.5
    # Reward for winning a battle
    battle_win: float = 10.0
    # Penalty for losing a battle
    battle_loss: float = -5.0

    # === Milestone Rewards ===
    # Reward per badge obtained
    milestone_badge: float = 100.0
    # Reward per Pokemon level gained
    milestone_level_up: float = 1.0
    # Reward for reaching key locations
    milestone_key_location: float = 5.0
    # Reward per event flag set
    milestone_event: float = 4.0

    # === Penalty Rewards ===
    # Small penalty per step (encourages faster solutions)
    penalty_step: float = -0.01
    # Penalty for walking into walls (no position change)
    penalty_wall: float = -0.1
    # Penalty for staying in same location too long
    penalty_stuck: float = -0.05

    # === Legacy Rewards (for backward compatibility) ===
    # Healing reward coefficient
    legacy_heal: float = 10.0
    # Global reward scale multiplier
    reward_scale: float = 1.0

    # === Reward Component Toggles ===
    # Enable/disable specific reward components
    enable_exploration: bool = True
    enable_battle: bool = True
    enable_milestone: bool = True
    enable_penalty: bool = True
    enable_legacy_heal: bool = True

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return asdict(self)

    def save_json(self, path: Path):
        """Save config to JSON file."""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'RewardConfig':
        """Create config from dictionary."""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    @classmethod
    def from_json(cls, path: Path) -> 'RewardConfig':
        """Load config from JSON file."""
        with open(path, 'r') as f:
            data = json.load(f)
        return cls.from_dict(data)


# Predefined reward configurations for different training scenarios

def get_default_config() -> RewardConfig:
    """Default balanced reward configuration."""
    return RewardConfig()


def get_exploration_focused_config() -> RewardConfig:
    """Configuration that heavily rewards exploration."""
    return RewardConfig(
        exploration_new_tile=2.0,
        exploration_recent_tile=0.5,
        battle_hp_delta=0.1,
        battle_win=5.0,
        milestone_badge=50.0,
        penalty_step=-0.005,
    )


def get_battle_focused_config() -> RewardConfig:
    """Configuration that heavily rewards battle performance."""
    return RewardConfig(
        exploration_new_tile=0.5,
        battle_hp_delta=2.0,
        battle_win=50.0,
        battle_loss=-10.0,
        milestone_level_up=5.0,
        penalty_step=-0.02,
    )


def get_milestone_focused_config() -> RewardConfig:
    """Configuration that heavily rewards achieving milestones."""
    return RewardConfig(
        exploration_new_tile=0.5,
        battle_hp_delta=0.2,
        battle_win=20.0,
        milestone_badge=200.0,
        milestone_level_up=10.0,
        milestone_key_location=20.0,
        milestone_event=10.0,
        penalty_step=-0.005,
    )


def get_minimal_penalty_config() -> RewardConfig:
    """Configuration with minimal penalties (for early training)."""
    return RewardConfig(
        penalty_step=0.0,
        penalty_wall=-0.01,
        penalty_stuck=-0.01,
    )


# Configuration registry for easy access
REWARD_CONFIGS = {
    'default': get_default_config,
    'exploration': get_exploration_focused_config,
    'battle': get_battle_focused_config,
    'milestone': get_milestone_focused_config,
    'minimal_penalty': get_minimal_penalty_config,
}


def get_reward_config(name: str = 'default') -> RewardConfig:
    """
    Get a predefined reward configuration by name.

    Args:
        name: Name of the configuration ('default', 'exploration', 'battle', 'milestone', 'minimal_penalty')

    Returns:
        RewardConfig instance
    """
    if name not in REWARD_CONFIGS:
        raise ValueError(f"Unknown reward config: {name}. Available: {list(REWARD_CONFIGS.keys())}")
    return REWARD_CONFIGS[name]()
