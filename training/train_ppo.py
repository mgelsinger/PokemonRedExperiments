import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional

from env.red_gym_env import RedGymEnv
from env.stream_agent_wrapper import StreamWrapper
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList

from training.tensorboard_callback import TensorboardCallback

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
}

GPU_PRESETS = {
    # roughly balanced for typical desktop GPUs; adjust as needed
    "small": {"num_envs": 8, "batch_size": 256},
    "medium": {"num_envs": 16, "batch_size": 512},
    "large": {"num_envs": 32, "batch_size": 1024},
}


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
    parser.add_argument("--preset", choices=list(GPU_PRESETS.keys()), default=None, help="GPU sizing preset.")
    parser.add_argument("--stream", action="store_true", default=True, help="Enable map streaming.")
    parser.add_argument("--no-stream", dest="stream", action="store_false", help="Disable map streaming.")
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging.")
    parser.add_argument("--wandb-project", type=str, default="pokemon-train", help="wandb project name.")
    parser.add_argument("--wandb-run-name", type=str, default=None, help="Optional wandb run name.")
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

    env_defaults = {**DEFAULT_ENV_CONFIG, **base_conf.get("env", {})}
    train_defaults = {**DEFAULT_TRAIN_CONFIG, **base_conf.get("train", {})}
    train_defaults = apply_preset(train_defaults, args.preset)

    num_envs = args.num_envs or train_defaults["num_envs"]
    total_multiplier = args.total_multiplier or train_defaults["total_multiplier"]
    if num_envs < 1:
        raise ValueError("--num-envs must be >= 1")

    # episode length mirrors the original v2 script
    ep_length = env_defaults["max_steps"]

    run_dir = args.output_dir / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    env_config = merge_env_config(env_defaults, args.rom, args.state, run_dir)

    # Build vectorized envs
    env = SubprocVecEnv([make_env(i, env_config, args.stream) for i in range(num_envs)])

    checkpoint_callback = CheckpointCallback(save_freq=ep_length // 2, save_path=str(run_dir), name_prefix="poke")
    callbacks = [checkpoint_callback, TensorboardCallback(run_dir)]

    file_name = ""  # override with a checkpoint to resume
    train_steps_batch = ep_length // max(num_envs, 1)

    if file_name and Path(f"{file_name}.zip").exists():
        print("\nloading checkpoint")
        model = PPO.load(file_name, env=env)
        model.n_steps = train_steps_batch
        model.n_envs = num_envs
        model.rollout_buffer.buffer_size = train_steps_batch
        model.rollout_buffer.n_envs = num_envs
        model.rollout_buffer.reset()
    else:
        model = PPO(
            "MultiInputPolicy",
            env,
            verbose=1,
            n_steps=train_steps_batch,
            batch_size=train_defaults.get("batch_size", 512),
            n_epochs=train_defaults.get("n_epochs", 1),
            gamma=train_defaults.get("gamma", 0.997),
            ent_coef=train_defaults.get("ent_coef", 0.01),
            tensorboard_log=str(run_dir),
        )

    # save run metadata for dashboards/comparisons
    env_config_for_log = env_config.copy()
    if isinstance(env_config_for_log.get("session_path"), Path):
        env_config_for_log["session_path"] = str(env_config_for_log["session_path"])
    metadata = {
        "run_name": args.run_name,
        "run_dir": str(run_dir),
        "env_config": env_config_for_log,
        "train_config": {
            "num_envs": num_envs,
            "total_multiplier": total_multiplier,
            "batch_size": train_defaults.get("batch_size", 512),
            "n_epochs": train_defaults.get("n_epochs", 1),
            "gamma": train_defaults.get("gamma", 0.997),
            "ent_coef": train_defaults.get("ent_coef", 0.01),
            "preset": args.preset,
        },
        "stream_enabled": args.stream,
        "wandb_enabled": args.wandb,
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

    if wandb_run:
        wandb_run.finish()
