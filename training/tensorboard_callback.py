import os
import json

from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import Image
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from einops import rearrange, reduce

def merge_dicts(dicts):
    sum_dict = {}
    count_dict = {}
    distrib_dict = {}

    for d in dicts:
        for k, v in d.items():
            if isinstance(v, (int, float)): 
                sum_dict[k] = sum_dict.get(k, 0) + v
                count_dict[k] = count_dict.get(k, 0) + 1
                distrib_dict.setdefault(k, []).append(v)

    mean_dict = {}
    for k in sum_dict:
        mean_dict[k] = sum_dict[k] / count_dict[k]
        distrib_dict[k] = np.array(distrib_dict[k])

    return mean_dict, distrib_dict

class TensorboardCallback(BaseCallback):

    def __init__(self, log_dir, verbose=0):
        super().__init__(verbose)
        self.log_dir = log_dir
        self.writer = None
        # Episode-level tracking
        self.episode_returns = []
        self.episode_lengths = []
        self.episode_successes = []
        self.episode_exploration_rewards = []
        self.episode_battle_rewards = []
        self.episode_milestone_rewards = []
        self.episode_penalty_rewards = []
        # Battle-focused tracking
        self.episode_battles_started = []
        self.episode_battles_won = []
        self.episode_battles_total = []
        self.episode_steps_to_first_battle = []
        # Milestone tracking
        self.episode_badges_earned = []
        self.episode_levels_gained = []
        self.episode_deaths = []
        self.episode_map_progress = []

    def _on_training_start(self):
        if self.writer is None:
            self.writer = SummaryWriter(log_dir=os.path.join(self.log_dir, 'histogram'))

    def _on_step(self) -> bool:
        # Check for episode completions from VecMonitor
        # VecMonitor adds 'episode' key to info dicts when episodes finish
        infos = self.locals.get('infos', [])
        for info in infos:
            if 'episode' in info:
                ep_info = info['episode']
                # Collect episode metrics
                self.episode_returns.append(ep_info['r'])
                self.episode_lengths.append(ep_info['l'])
                self.episode_exploration_rewards.append(ep_info.get('exploration_r', 0))
                self.episode_battle_rewards.append(ep_info.get('battle_r', 0))
                self.episode_milestone_rewards.append(ep_info.get('milestone_r', 0))
                self.episode_penalty_rewards.append(ep_info.get('penalty_r', 0))

                # Battle-focused metrics
                self.episode_battles_started.append(ep_info.get('battles_started', 0))
                self.episode_battles_won.append(ep_info.get('battles_won', 0))
                self.episode_battles_total.append(ep_info.get('battles_total', 0))
                self.episode_steps_to_first_battle.append(ep_info.get('steps_to_first_battle', ep_info['l']))

                # Milestone metrics
                self.episode_badges_earned.append(ep_info.get('badges_earned', 0))
                self.episode_levels_gained.append(ep_info.get('levels_gained', 0))
                self.episode_deaths.append(ep_info.get('deaths', 0))
                self.episode_map_progress.append(ep_info.get('map_progress_max', 0))

                # Track success if present
                if 'success' in info:
                    self.episode_successes.append(1.0 if info['success'] else 0.0)

        # Log episode-level metrics when we have data
        if len(self.episode_returns) > 0:
            self.logger.record("train/episode_return_mean", np.mean(self.episode_returns))
            self.logger.record("train/episode_return_max", np.max(self.episode_returns))
            self.logger.record("train/episode_return_min", np.min(self.episode_returns))
            self.logger.record("train/episode_length_mean", np.mean(self.episode_lengths))
            self.logger.record("train/exploration_return", np.mean(self.episode_exploration_rewards))
            self.logger.record("train/battle_return", np.mean(self.episode_battle_rewards))
            self.logger.record("train/milestone_return", np.mean(self.episode_milestone_rewards))
            self.logger.record("train/penalty_return", np.mean(self.episode_penalty_rewards))

            # Battle-focused metrics
            self.logger.record("train/episode_battles_started_mean", np.mean(self.episode_battles_started))
            self.logger.record("train/episode_battles_won_mean", np.mean(self.episode_battles_won))
            self.logger.record("train/episode_battles_total_mean", np.mean(self.episode_battles_total))
            if len(self.episode_steps_to_first_battle) > 0:
                valid_steps = [s for s in self.episode_steps_to_first_battle if s is not None]
                if valid_steps:
                    self.logger.record("train/episode_steps_to_first_battle", np.mean(valid_steps))

            # Milestone metrics
            self.logger.record("train/episode_badges_earned_mean", np.mean(self.episode_badges_earned))
            self.logger.record("train/episode_levels_gained_mean", np.mean(self.episode_levels_gained))
            self.logger.record("train/episode_deaths_mean", np.mean(self.episode_deaths))
            self.logger.record("train/episode_map_progress_max", np.mean(self.episode_map_progress))

            if len(self.episode_successes) > 0:
                self.logger.record("train/success_rate", np.mean(self.episode_successes))

            # Write histograms
            self.writer.add_histogram("train/episode_return_distrib", np.array(self.episode_returns), self.n_calls)
            self.writer.add_histogram("train/episode_length_distrib", np.array(self.episode_lengths), self.n_calls)

            # Clear buffers after logging
            self.episode_returns.clear()
            self.episode_lengths.clear()
            self.episode_successes.clear()
            self.episode_exploration_rewards.clear()
            self.episode_battle_rewards.clear()
            self.episode_milestone_rewards.clear()
            self.episode_penalty_rewards.clear()
            self.episode_battles_started.clear()
            self.episode_battles_won.clear()
            self.episode_battles_total.clear()
            self.episode_steps_to_first_battle.clear()
            self.episode_badges_earned.clear()
            self.episode_levels_gained.clear()
            self.episode_deaths.clear()
            self.episode_map_progress.clear()

        # Check if any environment has reached max_steps (episode ended)
        step_counts = self.training_env.get_attr("step_count")
        max_steps = self.training_env.get_attr("max_steps")
        if step_counts[0] >= max_steps[0] - 1:
            all_infos = self.training_env.get_attr("agent_stats")
            all_final_infos = [stats[-1] for stats in all_infos]
            mean_infos, distributions = merge_dicts(all_final_infos)
            # TODO log distributions, and total return
            for key, val in mean_infos.items():
                self.logger.record(f"env_stats/{key}", val)

            for key, distrib in distributions.items():
                self.writer.add_histogram(f"env_stats_distribs/{key}", distrib, self.n_calls)
                self.logger.record(f"env_stats_max/{key}", max(distrib))

            # === NEW: Log reward component breakdown ===
            # Get episode reward components from all environments
            episode_reward_components = self.training_env.get_attr("episode_reward_components")

            # Compute mean and distribution for each component
            component_means = {}
            component_distribs = {}
            for component in ['exploration', 'battle', 'milestone', 'penalty', 'legacy']:
                values = [env_comps[component] for env_comps in episode_reward_components]
                component_means[component] = np.mean(values)
                component_distribs[component] = np.array(values)

            # Log mean values for each reward component
            for component, mean_val in component_means.items():
                self.logger.record(f"reward_components/{component}", mean_val)

            # Log total episode return (sum of all shaped components, excluding legacy)
            total_shaped_rewards = [
                comps['exploration'] + comps['battle'] + comps['milestone'] + comps['penalty']
                for comps in episode_reward_components
            ]
            self.logger.record("reward_components/total_shaped", np.mean(total_shaped_rewards))
            self.logger.record("reward_components/total_shaped_max", np.max(total_shaped_rewards))
            self.logger.record("reward_components/total_shaped_min", np.min(total_shaped_rewards))

            # Log histograms for reward components
            for component, distrib in component_distribs.items():
                self.writer.add_histogram(f"reward_distribs/{component}", distrib, self.n_calls)

            # Log episode lengths
            episode_lengths = [len(stats) for stats in all_infos]
            self.logger.record("episode/length_mean", np.mean(episode_lengths))
            self.logger.record("episode/length_max", np.max(episode_lengths))
            self.logger.record("episode/length_min", np.min(episode_lengths))
            self.writer.add_histogram("episode/length_distrib", np.array(episode_lengths), self.n_calls)

            # === NEW: Log battle statistics ===
            episode_battle_stats = self.training_env.get_attr("episode_battle_stats")
            battles_won = [stats['battles_won'] for stats in episode_battle_stats]
            battles_lost = [stats['battles_lost'] for stats in episode_battle_stats]
            battles_total = [stats['battles_total'] for stats in episode_battle_stats]

            # Win rate (avoid division by zero)
            win_rates = [
                won / total if total > 0 else 0.0
                for won, total in zip(battles_won, battles_total)
            ]

            self.logger.record("battle_stats/wins_mean", np.mean(battles_won))
            self.logger.record("battle_stats/losses_mean", np.mean(battles_lost))
            self.logger.record("battle_stats/total_mean", np.mean(battles_total))
            self.logger.record("battle_stats/win_rate_mean", np.mean(win_rates))

            if len(battles_won) > 0:
                self.writer.add_histogram("battle_stats/wins_distrib", np.array(battles_won), self.n_calls)
                self.writer.add_histogram("battle_stats/win_rate_distrib", np.array(win_rates), self.n_calls)

            #images = self.training_env.get_attr("recent_screens")
            #images_row = rearrange(np.array(images), "(r f) h w c -> (r c h) (f w)", r=2)
            #self.logger.record("trajectory/image", Image(images_row, "HW"), exclude=("stdout", "log", "json", "csv"))

            explore_map = np.array(self.training_env.get_attr("explore_map"))
            map_sum = reduce(explore_map, "f h w -> h w", "max")
            self.logger.record("trajectory/explore_sum", Image(map_sum, "HW"), exclude=("stdout", "log", "json", "csv"))

            map_row = rearrange(explore_map, "(r f) h w -> (r h) (f w)", r=2)
            self.logger.record("trajectory/explore_map", Image(map_row, "HW"), exclude=("stdout", "log", "json", "csv"))

            list_of_flag_dicts = self.training_env.get_attr("current_event_flags_set")
            merged_flags = {k: v for d in list_of_flag_dicts for k, v in d.items()}
            self.logger.record("trajectory/all_flags", json.dumps(merged_flags))

        return True
    
    def _on_training_end(self):
        if self.writer:
            self.writer.close()

