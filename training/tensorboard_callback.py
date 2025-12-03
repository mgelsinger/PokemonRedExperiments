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

    def _on_training_start(self):
        if self.writer is None:
            self.writer = SummaryWriter(log_dir=os.path.join(self.log_dir, 'histogram'))

    def _on_step(self) -> bool:

        if self.training_env.env_method("check_if_done", indices=[0])[0]:
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

