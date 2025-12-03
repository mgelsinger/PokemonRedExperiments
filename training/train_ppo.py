import argparse
import json
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from env.red_gym_env import RedGymEnv
from env.stream_agent_wrapper import StreamWrapper
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList

from training.tensorboard_callback import TensorboardCallback
from training.config_utils import validate_env_config, validate_train_config
from training.status_tracking import StatusWriterCallback, PeriodicEvalCallback

DEFAULT_CONFIG_PATH = Path("configs") / "train_default.json"

DEFAULT_ENV_CONFIG: Dict[str, Any] = {
    "headless": True,
    "save_final_state": False,
    "early_stop": False,
    "action_freq": 24,
    "max_steps": 2048 * 80,
    "print_rewards": True,
    "save_video": False,
    "fast_video": True,
    "debug": False,
    "reward_scale": 0.5,
    "explore_weight": 0.25,
}

DEFAULT_TRAIN_CONFIG: Dict[str, Any] = {
    "num_envs": 16,
    "total_multiplier": 10000,
    "batch_size": 512,
    "n_epochs": 1,
    "gamma": 0.997,
    "ent_coef": 0.01,
    "learning_rate": 3e-4,
    "clip_range": 0.2,
    "vf_coef": 0.5,
    "max_grad_norm": 0.5,
    "gae_lambda": 0.95,
    "use_sde": False,
    "sde_sample_freq": -1,
    "normalize_advantage": True,
    "target_kl": None,
}

GPU_PRESETS = {
    # roughly balanced for typical desktop GPUs; adjust as needed
    "small": {"num_envs": 8, "batch_size": 256},
    "medium": {"num_envs": 16, "batch_size": 512},
    "large": {"num_envs": 32, "batch_size": 1024},
}


def seed_everything(seed: Optional[int]):
    if seed is None:
        return
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except Exception:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except Exception:
        pass
    set_random_seed(seed)


def get_git_commit(repo_root: Path) -> Optional[str]:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=repo_root, capture_output=True, text=True, check=True
        )
        return result.stdout.strip()
    except Exception:
        return None


def find_latest_checkpoint(run_dir: Path) -> Optional[Tuple[Path, float]]:
    zip_files = list(run_dir.glob("**/*.zip"))
    if not zip_files:
        return None
    most_recent = max(zip_files, key=lambda p: p.stat().st_mtime)
    age_hours = (time.time() - most_recent.stat().st_mtime) / 3600
    return most_recent, age_hours


def load_config(config_path: Path) -> Dict[str, Any]:
    if not config_path.exists():
        return {}
    with config_path.open() as f:
        return json.load(f)


def merge_env_config(base_env: Dict[str, Any], rom: Path, state: Path, session_path: Path) -> Dict[str, Any]:
    env_conf = base_env.copy()
    env_conf["gb_path"] = str(rom)
    env_conf["init_state"] = str(state)
    env_conf["session_path"] = session_path
    env_conf.setdefault("headless", True)
    env_conf.setdefault("save_video", False)
    env_conf.setdefault("fast_video", True)
    return env_conf


def make_env(rank: int, env_conf: Dict[str, Any], stream: bool, seed: int = 0):
    """
    Utility function for multiprocessed env.
    :param rank: (int) index of the subprocess
    """

    def _init():
        env = RedGymEnv(env_conf)
        if stream:
            env = StreamWrapper(
                env,
                stream_metadata={
                    "user": "v2-default",
                    "env_id": rank,
                    "color": "#447799",
                    "extra": "",
                },
            )
        env.reset(seed=(seed + rank))
        return env

    set_random_seed(seed)
    return _init


def parse_args():
    parser = argparse.ArgumentParser(description="Train PPO agent on Pokemon Red (V2 env).")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH, help="Path to train config JSON.")
    parser.add_argument("--rom", type=Path, default=Path("PokemonRed.gb"), help="Path to Pokemon Red ROM.")
    parser.add_argument("--state", type=Path, default=Path("init.state"), help="Initial save state path.")
    parser.add_argument("--run-name", type=str, default="poke_run", help="Run name for outputs.")
    parser.add_argument("--output-dir", type=Path, default=Path("runs"), help="Directory to store checkpoints/logs.")
    parser.add_argument("--num-envs", type=int, default=None, help="Override number of parallel envs.")
    parser.add_argument(
        "--total-multiplier",
        type=int,
        default=None,
        help="Multiplier for total timesteps (ep_length * num_envs * total_multiplier).",
    )
    parser.add_argument("--batch-size", type=int, default=None, help="PPO minibatch size.")
    parser.add_argument("--preset", choices=list(GPU_PRESETS.keys()), default=None, help="GPU sizing preset.")
    parser.add_argument("--stream", action="store_true", default=True, help="Enable map streaming.")
    parser.add_argument("--no-stream", dest="stream", action="store_false", help="Disable map streaming.")
    parser.add_argument("--checkpoint-freq", type=int, default=None, help="Steps between checkpoints (default: max_steps/2).")
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging.")
    parser.add_argument("--wandb-project", type=str, default="pokemon-train", help="wandb project name.")
    parser.add_argument("--wandb-run-name", type=str, default=None, help="Optional wandb run name.")
    parser.add_argument("--seed", type=int, default=None, help="Global random seed.")
    parser.add_argument("--resume-checkpoint", type=Path, default=None, help="Resume from a specific checkpoint .zip.")
    parser.add_argument("--resume-latest", action="store_true", help="Resume from the latest checkpoint under the run directory.")
    parser.add_argument("--status-file", type=Path, default=None, help="Path to write status.json (default: runs/<run>/status.json).")
    parser.add_argument("--status-interval", type=float, default=10.0, help="Seconds between status snapshots.")
    parser.add_argument("--eval-log", type=Path, default=None, help="Path to eval jsonl (default: runs/<run>/eval.jsonl).")
    parser.add_argument("--eval-every-steps", type=int, default=None, help="Run eval every N training timesteps (disabled if not set).")
    parser.add_argument("--eval-episodes", type=int, default=2, help="Number of episodes per eval run.")
    parser.add_argument(
        "--eval-max-steps", type=int, default=None, help="Max steps per eval episode (default: env max_steps)."
    )
    parser.add_argument("--eval-stream", action="store_true", help="Enable streaming overlays during eval.")
    parser.add_argument("--no-eval", dest="eval_enabled", action="store_false", help="Disable periodic eval.")
    parser.set_defaults(eval_enabled=True)
    return parser.parse_args()


def validate_paths(rom: Path, state: Path):
    missing = []
    if not rom.exists():
        missing.append(f"ROM not found at {rom}")
    if not state.exists():
        missing.append(f"State file not found at {state}")
    if missing:
        raise FileNotFoundError("\n".join(missing))


def apply_preset(train_conf: Dict[str, Any], preset: Optional[str]) -> Dict[str, Any]:
    if preset and preset in GPU_PRESETS:
        return {**train_conf, **GPU_PRESETS[preset]}
    return train_conf


if __name__ == "__main__":
    args = parse_args()
    base_conf = load_config(args.config)

    validate_paths(args.rom, args.state)
    seed_everything(args.seed)
    git_commit = get_git_commit(REPO_ROOT)

    env_defaults = {**DEFAULT_ENV_CONFIG, **base_conf.get("env", {})}
    train_defaults = {**DEFAULT_TRAIN_CONFIG, **base_conf.get("train", {})}
    train_defaults = apply_preset(train_defaults, args.preset)

    num_envs = args.num_envs or train_defaults["num_envs"]
    total_multiplier = args.total_multiplier or train_defaults["total_multiplier"]
    batch_size = args.batch_size or train_defaults["batch_size"]

    # Extract all PPO hyperparameters
    n_epochs = train_defaults.get("n_epochs", 1)
    gamma = train_defaults.get("gamma", 0.997)
    ent_coef = train_defaults.get("ent_coef", 0.01)
    learning_rate = train_defaults.get("learning_rate", 3e-4)
    clip_range = train_defaults.get("clip_range", 0.2)
    vf_coef = train_defaults.get("vf_coef", 0.5)
    max_grad_norm = train_defaults.get("max_grad_norm", 0.5)
    gae_lambda = train_defaults.get("gae_lambda", 0.95)
    use_sde = train_defaults.get("use_sde", False)
    sde_sample_freq = train_defaults.get("sde_sample_freq", -1)
    normalize_advantage = train_defaults.get("normalize_advantage", True)
    target_kl = train_defaults.get("target_kl", None)

    train_config_resolved = {
        **train_defaults,
        "num_envs": num_envs,
        "total_multiplier": total_multiplier,
        "batch_size": batch_size,
        "n_epochs": n_epochs,
        "gamma": gamma,
        "ent_coef": ent_coef,
        "learning_rate": learning_rate,
        "clip_range": clip_range,
        "vf_coef": vf_coef,
        "max_grad_norm": max_grad_norm,
        "gae_lambda": gae_lambda,
        "use_sde": use_sde,
        "sde_sample_freq": sde_sample_freq,
        "normalize_advantage": normalize_advantage,
        "target_kl": target_kl,
    }
    if batch_size < 1:
        raise ValueError("--batch-size must be >= 1")
    if num_envs < 1:
        raise ValueError("--num-envs must be >= 1")

    env_config = merge_env_config(env_defaults, args.rom, args.state, args.output_dir / args.run_name)

    env_errors, env_warnings = validate_env_config(env_config)
    train_errors, train_warnings = validate_train_config(train_config_resolved)
    all_errors = env_errors + train_errors
    all_warnings = env_warnings + train_warnings
    if all_warnings:
        print("\nWarnings:")
        for w in all_warnings:
            print(f" - {w}")
    if all_errors:
        raise ValueError("Config validation failed:\n" + "\n".join(f" - {e}" for e in all_errors))

    # episode length mirrors the original v2 script
    ep_length = env_defaults["max_steps"]
    total_timesteps_target = ep_length * num_envs * total_multiplier

    run_dir = args.output_dir / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    env_config["session_path"] = run_dir

    status_path = args.status_file or (run_dir / "status.json")
    eval_log_path = args.eval_log or (run_dir / "eval.jsonl")
    eval_every_steps = args.eval_every_steps if args.eval_enabled else None
    eval_episodes = max(args.eval_episodes, 1)
    eval_max_steps = args.eval_max_steps or env_defaults["max_steps"]
    if eval_max_steps is not None and eval_max_steps < 1:
        raise ValueError("--eval-max-steps must be >= 1 when provided")

    # Build vectorized envs
    env = SubprocVecEnv([make_env(i, env_config, args.stream, seed=args.seed or 0) for i in range(num_envs)])

    ckpt_freq = args.checkpoint_freq or (ep_length // 2)
    checkpoint_callback = CheckpointCallback(save_freq=ckpt_freq, save_path=str(run_dir), name_prefix="poke")

    status_callback = StatusWriterCallback(
        status_path=status_path,
        total_timesteps=total_timesteps_target,
        run_name=args.run_name,
        train_config=train_config_resolved,
        env_config=env_config,
        interval_seconds=args.status_interval,
    )
    callbacks = [checkpoint_callback, TensorboardCallback(run_dir), status_callback]

    if eval_every_steps:
        eval_env_conf = env_config.copy()
        eval_env_conf.update(
            {
                "session_path": run_dir / "eval_session",
                "headless": True,
                "save_video": False,
                "fast_video": True,
                "print_rewards": False,
                "gb_path": str(args.rom),
                "init_state": str(args.state),
            }
        )

        def make_eval_env():
            base_env = RedGymEnv(eval_env_conf)
            if args.eval_stream:
                return StreamWrapper(
                    base_env,
                    stream_metadata={
                        "user": "eval",
                        "env_id": 9999,
                        "color": "#995533",
                        "extra": "eval",
                    },
                )
            return base_env

        eval_callback = PeriodicEvalCallback(
            eval_env_fn=make_eval_env,
            eval_log_path=eval_log_path,
            eval_every_steps=eval_every_steps,
            eval_episodes=eval_episodes,
            eval_max_steps=eval_max_steps,
            status_callback=status_callback,
            base_eval_seed=(args.seed or 0) + 1234,
        )
        callbacks.append(eval_callback)

    resume_checkpoint = args.resume_checkpoint
    resume_source = None
    if args.resume_latest and resume_checkpoint is None:
        latest = find_latest_checkpoint(run_dir)
        if latest is not None:
            resume_checkpoint, age_hours = latest
            resume_source = f"latest (age {age_hours:.2f}h)"
            print(f"Resuming from latest checkpoint: {resume_checkpoint} (age {age_hours:.2f}h)")
        else:
            print("No checkpoint found for --resume-latest; starting fresh.")
    if resume_checkpoint and resume_checkpoint.suffix != ".zip":
        resume_checkpoint = resume_checkpoint.with_suffix(".zip")

    train_steps_batch = ep_length // max(num_envs, 1)
    if resume_checkpoint and resume_checkpoint.exists():
        if resume_source is None:
            resume_source = str(resume_checkpoint)
        print("\nloading checkpoint")
        model = PPO.load(str(resume_checkpoint), env=env)
        model.n_steps = train_steps_batch
        model.n_envs = num_envs
        model.rollout_buffer.buffer_size = train_steps_batch
        model.rollout_buffer.n_envs = num_envs
        model.rollout_buffer.reset()
    else:
        if resume_checkpoint:
            print(f"Requested resume checkpoint not found: {resume_checkpoint} (starting fresh)")
        model = PPO(
            "MultiInputPolicy",
            env,
            verbose=1,
            n_steps=train_steps_batch,
            batch_size=batch_size,
            n_epochs=n_epochs,
            gamma=gamma,
            ent_coef=ent_coef,
            learning_rate=learning_rate,
            clip_range=clip_range,
            vf_coef=vf_coef,
            max_grad_norm=max_grad_norm,
            gae_lambda=gae_lambda,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            normalize_advantage=normalize_advantage,
            target_kl=target_kl,
            tensorboard_log=str(run_dir),
        )

    # save run metadata for dashboards/comparisons
    env_config_for_log = env_config.copy()
    if isinstance(env_config_for_log.get("session_path"), Path):
        env_config_for_log["session_path"] = str(env_config_for_log["session_path"])
    metadata = {
        "run_name": args.run_name,
        "run_dir": str(run_dir),
        "git_commit": git_commit,
        "seed": args.seed,
        "env_config": env_config_for_log,
        "train_config": {
            **train_config_resolved,
            "preset": args.preset,
        },
        "stream_enabled": args.stream,
        "wandb_enabled": args.wandb,
        "status_file": str(status_path),
        "status_interval_seconds": args.status_interval,
        "total_timesteps_target": total_timesteps_target,
        "eval": {
            "enabled": bool(eval_every_steps),
            "log": str(eval_log_path),
            "every_steps": eval_every_steps,
            "episodes": eval_episodes,
            "max_steps": eval_max_steps,
            "stream": args.eval_stream,
        },
        "resume_from": resume_source,
    }
    (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))

    # Optional wandb
    wandb_run = None
    if args.wandb:
        import wandb
        from wandb.integration.sb3 import WandbCallback

        wandb.tensorboard.patch(root_logdir=str(run_dir))
        wandb_run = wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name or args.run_name,
            config=metadata,
            sync_tensorboard=True,
            monitor_gym=True,
            save_code=True,
        )
        callbacks.append(WandbCallback())

    print(model.policy)

    model.learn(
        total_timesteps=ep_length * num_envs * total_multiplier,
        callback=CallbackList(callbacks),
        tb_log_name="poke_ppo",
    )

    # always save final snapshot
    final_path = run_dir / "final.zip"
    model.save(str(final_path))

    if wandb_run:
        wandb_run.finish()
