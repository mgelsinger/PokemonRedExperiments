import json
import os
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


def convert_numpy_types(obj: Any) -> Any:
    """
    Recursively convert numpy types to native Python types for JSON serialization.
    """
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_numpy_types(item) for item in obj]
    else:
        return obj


def atomic_write_json(path: Path, payload: Dict[str, Any]):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    # Convert numpy types before serialization
    clean_payload = convert_numpy_types(payload)
    tmp.write_text(json.dumps(clean_payload, indent=2))
    tmp.replace(path)


def append_jsonl(path: Path, payload: Dict[str, Any]):
    path.parent.mkdir(parents=True, exist_ok=True)
    # Convert numpy types before serialization
    clean_payload = convert_numpy_types(payload)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(clean_payload) + os.linesep)


class StatusWriterCallback(BaseCallback):
    """
    Periodically writes a status.json snapshot for the UI/monitoring.
    """

    def __init__(
        self,
        status_path: Path,
        total_timesteps: int,
        run_name: str,
        train_config: Dict[str, Any],
        env_config: Dict[str, Any],
        interval_seconds: float = 10.0,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.status_path = status_path
        self.total_timesteps = total_timesteps
        self.run_name = run_name
        self.train_config = train_config
        self.env_config = env_config
        self.interval_seconds = max(interval_seconds, 1.0)

        self._start_time: float = 0.0
        self._last_write_time: float = 0.0
        self._last_timesteps: int = 0
        self._last_eval: Optional[Dict[str, Any]] = None

    def _on_training_start(self) -> None:
        now = time.time()
        self._start_time = now
        self._last_write_time = now
        self._last_timesteps = 0
        self._write_status(now, status="running")

    def _on_step(self) -> bool:
        now = time.time()
        if now - self._last_write_time < self.interval_seconds:
            return True
        self._write_status(now, status="running")
        return True

    def _on_training_end(self) -> None:
        self._write_status(time.time(), status="finished", force=True)

    def record_eval_result(self, eval_result: Dict[str, Any]) -> None:
        self._last_eval = eval_result
        self._write_status(time.time(), status="running", force=True)

    def _extract_metrics(self) -> Dict[str, Any]:
        # SB3 logger stores latest values in name_to_value; keep only a few useful ones.
        logger_values = getattr(self.logger, "name_to_value", {}) or {}
        keys = [
            "rollout/ep_rew_mean",
            "rollout/ep_len_mean",
            "train/value_loss",
            "train/policy_gradient_loss",
            "train/entropy_loss",
            "train/approx_kl",
            "train/clip_fraction",
            "train/loss",
            "train/learning_rate",
        ]
        return {k: logger_values.get(k) for k in keys if k in logger_values}

    def _write_status(self, now: float, status: str, force: bool = False) -> None:
        elapsed = now - self._last_write_time
        timesteps = int(self.num_timesteps)
        if not force and elapsed < self.interval_seconds:
            return

        steps_delta = max(timesteps - self._last_timesteps, 0)
        throughput = steps_delta / elapsed if elapsed > 0 else None

        payload = {
            "run_name": self.run_name,
            "status": status,
            "timesteps_done": timesteps,
            "timesteps_total": self.total_timesteps,
            "progress": (timesteps / self.total_timesteps) if self.total_timesteps else None,
            "num_envs": self.train_config.get("num_envs"),
            "batch_size": self.train_config.get("batch_size"),
            "n_epochs": self.train_config.get("n_epochs"),
            "gamma": self.train_config.get("gamma"),
            "ent_coef": self.train_config.get("ent_coef"),
            "reward_scale": self.env_config.get("reward_scale"),
            "explore_weight": self.env_config.get("explore_weight"),
            "start_time": self._start_time,
            "last_write_time": now,
            "wall_clock_seconds": now - self._start_time,
            "throughput_steps_per_sec": throughput,
            "latest_metrics": self._extract_metrics(),
        }

        if self._last_eval:
            payload["last_eval"] = self._last_eval

        atomic_write_json(self.status_path, payload)
        self._last_write_time = now
        self._last_timesteps = timesteps


class PeriodicEvalCallback(BaseCallback):
    """
    Runs deterministic evaluations at a fixed step cadence and logs results to JSONL.
    """

    def __init__(
        self,
        eval_env_fn: Callable[[], Any],
        eval_log_path: Path,
        eval_every_steps: Optional[int],
        eval_episodes: int,
        eval_max_steps: Optional[int],
        status_callback: Optional[StatusWriterCallback],
        base_eval_seed: int = 0,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.eval_env_fn = eval_env_fn
        self.eval_log_path = eval_log_path
        self.eval_every_steps = eval_every_steps
        self.eval_episodes = max(eval_episodes, 1)
        self.eval_max_steps = eval_max_steps
        self.status_callback = status_callback
        self.base_eval_seed = base_eval_seed

        self._last_eval_step: int = 0
        self._eval_env = None

    def _on_training_start(self) -> None:
        # lazily instantiate eval env to avoid issues before model is ready
        pass

    def _on_step(self) -> bool:
        if not self.eval_every_steps:
            return True
        if (self.num_timesteps - self._last_eval_step) < self.eval_every_steps:
            return True

        try:
            result = self._run_eval()
        except Exception as exc:
            if self.verbose:
                print(f"[eval] failed: {exc}")
            self._last_eval_step = self.num_timesteps
            return True
        self._last_eval_step = self.num_timesteps
        append_jsonl(self.eval_log_path, result)
        if self.status_callback:
            self.status_callback.record_eval_result(result)
        return True

    def _on_training_end(self) -> None:
        if self._eval_env:
            try:
                self._eval_env.close()
            except Exception:
                pass

    def _ensure_env(self):
        if self._eval_env is None:
            self._eval_env = self.eval_env_fn()
        return self._eval_env

    def _run_eval(self) -> Dict[str, Any]:
        env = self._ensure_env()
        rewards: List[float] = []
        lengths: List[int] = []
        max_steps = self.eval_max_steps
        timestamp = time.time()

        # Battle and milestone tracking
        battles_started_list: List[int] = []
        battles_won_list: List[int] = []
        badges_earned_list: List[int] = []
        levels_gained_list: List[int] = []
        successes: List[bool] = []

        for idx in range(self.eval_episodes):
            seed = self.base_eval_seed + idx
            obs, info = env.reset(seed=seed)
            done = False
            truncated = False
            total_r = 0.0
            steps = 0
            while not done and not truncated:
                action, _states = self.model.predict(obs, deterministic=True)
                obs, reward, done, truncated, info = env.step(action)
                total_r += float(reward)
                steps += 1
                if max_steps and steps >= max_steps:
                    truncated = True

            rewards.append(total_r)
            lengths.append(steps)

            # Extract battle/milestone metrics from final info
            if 'episode' in info:
                ep = info['episode']
                battles_started_list.append(ep.get('battles_started', 0))
                battles_won_list.append(ep.get('battles_won', 0))
                badges_earned_list.append(ep.get('badges_earned', 0))
                levels_gained_list.append(ep.get('levels_gained', 0))

            successes.append(info.get('success', False))

        # Compute means
        mean_battles_started = sum(battles_started_list) / len(battles_started_list) if battles_started_list else 0.0
        mean_battles_won = sum(battles_won_list) / len(battles_won_list) if battles_won_list else 0.0
        mean_badges = sum(badges_earned_list) / len(badges_earned_list) if badges_earned_list else 0.0
        mean_levels = sum(levels_gained_list) / len(levels_gained_list) if levels_gained_list else 0.0
        success_rate = sum(successes) / len(successes) if successes else 0.0

        return {
            "timestamp": timestamp,
            "timesteps_when_ran": int(self.num_timesteps),
            "episodes": self.eval_episodes,
            "mean_reward": sum(rewards) / len(rewards),
            "mean_length": sum(lengths) / len(lengths),
            "rewards": rewards,
            "lengths": lengths,
            # Battle metrics
            "mean_battles_started": mean_battles_started,
            "mean_battles_won": mean_battles_won,
            "mean_badges_earned": mean_badges,
            "mean_levels_gained": mean_levels,
            "success_rate": success_rate,
            # Detail arrays
            "battles_started": battles_started_list,
            "battles_won": battles_won_list,
            "badges_earned": badges_earned_list,
            "levels_gained": levels_gained_list,
        }
