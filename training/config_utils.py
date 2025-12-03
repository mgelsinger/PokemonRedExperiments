import multiprocessing
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Tuple


def _ensure_positive_int(value: Any, name: str, errors: List[str]):
    if not isinstance(value, int) or value < 1:
        errors.append(f"{name} must be a positive integer (got {value})")


def validate_env_config(env_conf: Dict[str, Any]) -> Tuple[List[str], List[str]]:
    errors: List[str] = []
    warnings: List[str] = []

    _ensure_positive_int(env_conf.get("action_freq"), "action_freq", errors)
    _ensure_positive_int(env_conf.get("max_steps"), "max_steps", errors)

    reward_scale = env_conf.get("reward_scale")
    if reward_scale is not None and reward_scale <= 0:
        warnings.append(f"reward_scale is non-positive ({reward_scale}); training may be unstable.")

    explore_weight = env_conf.get("explore_weight")
    if explore_weight is not None and explore_weight < 0:
        warnings.append("explore_weight is negative; exploration bonuses will penalize the agent.")

    return errors, warnings


def validate_train_config(train_conf: Dict[str, Any]) -> Tuple[List[str], List[str]]:
    errors: List[str] = []
    warnings: List[str] = []

    num_envs = train_conf.get("num_envs")
    batch_size = train_conf.get("batch_size")
    total_multiplier = train_conf.get("total_multiplier")
    n_epochs = train_conf.get("n_epochs")
    gamma = train_conf.get("gamma")
    ent_coef = train_conf.get("ent_coef")

    _ensure_positive_int(num_envs, "num_envs", errors)
    _ensure_positive_int(batch_size, "batch_size", errors)
    _ensure_positive_int(total_multiplier, "total_multiplier", errors)
    _ensure_positive_int(n_epochs, "n_epochs", errors)

    if isinstance(batch_size, int) and isinstance(num_envs, int) and batch_size % num_envs != 0:
        errors.append(f"batch_size ({batch_size}) must be divisible by num_envs ({num_envs}) for even minibatches.")

    if gamma is not None and not (0 < gamma <= 1):
        errors.append(f"gamma must be in (0, 1]; got {gamma}")
    if ent_coef is not None and ent_coef < 0:
        errors.append(f"ent_coef must be >= 0; got {ent_coef}")

    try:
        cpu_count = multiprocessing.cpu_count()
        if isinstance(num_envs, int) and num_envs > cpu_count:
            warnings.append(f"num_envs ({num_envs}) exceeds CPU count ({cpu_count}); consider lowering for stability.")
    except NotImplementedError:
        pass

    return errors, warnings


def merge_metadata(env_conf: Dict[str, Any], train_conf: Dict[str, Any]) -> Dict[str, Any]:
    """Simple helper to produce a serializable copy for logging."""
    env_meta = {}
    for k, v in env_conf.items():
        if isinstance(v, Path):
            env_meta[k] = str(v)
        else:
            env_meta[k] = v

    train_meta = {}
    for k, v in train_conf.items():
        if isinstance(v, Path):
            train_meta[k] = str(v)
        else:
            train_meta[k] = v

    return {"env_config": env_meta, "train_config": train_meta}
