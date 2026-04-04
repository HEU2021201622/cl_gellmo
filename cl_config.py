import copy
from pathlib import Path
from typing import Any, Dict, Iterable

import yaml

from config import CL_DEFAULT_BASE_MODEL, CL_DEFAULT_LORA_TARGET_MODULES


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r") as handle:
        data = yaml.safe_load(handle) or {}
    return data


def load_cl_config(config_path: str) -> Dict[str, Any]:
    config_file = Path(config_path)
    config = load_yaml(str(config_file))

    base_config_path = config.get("base_config")
    if base_config_path:
        base_file = (config_file.parent / base_config_path).resolve()
        base_config = load_yaml(str(base_file))
        config = _deep_merge(base_config, config)

    config.setdefault("base_model", CL_DEFAULT_BASE_MODEL)
    config.setdefault("prompt_template_name", "mistral")
    config.setdefault("opt_type", "simple")
    config.setdefault("lora_target_modules", CL_DEFAULT_LORA_TARGET_MODULES)
    config.setdefault("learning_rate", 1e-4)
    config.setdefault("num_epochs", 1)
    config.setdefault("batch_size", 128)
    config.setdefault("micro_batch_size", 1)
    config.setdefault("cutoff_len", 4096)
    config.setdefault("warmup_ratio", 0.1)
    config.setdefault("train_on_inputs", False)
    config.setdefault("add_eos_token", False)
    config.setdefault("group_by_length", False)
    config.setdefault("gradient_checkpointing", False)
    config.setdefault("lr_scheduler", "cosine")
    config.setdefault("optim", "adamw_torch")
    config.setdefault("seed", 42)
    config.setdefault("data_root", "data")
    config.setdefault("train_file", "all_data/train.jsonl")
    config.setdefault("val_file", None)
    config.setdefault("buffer_capacity", 0)
    config.setdefault("replay_ratio", 0.0)
    config.setdefault("buffer_update_strategy", "task-balanced")
    config.setdefault("new_limit", 20000)
    config.setdefault("combo_limit", 2000)
    config.setdefault("val_limit", 1000)
    config.setdefault("save_step_records", True)
    config.setdefault("wandb_project", "")
    config.setdefault("wandb_run_name", "")
    config.setdefault("wandb_watch", "")
    config.setdefault("wandb_log_model", "")
    config.setdefault("load_in_8bit", False)

    method = config.get("method")
    if not method:
        raise ValueError("CL config must define `method`.")

    if method in {"naive_sequence", "replay"}:
        steps = config.get("steps") or {}
        if not steps:
            raise ValueError(f"{method} config must define `steps`.")
    elif method == "joint_all":
        joint_tasks = config.get("joint_tasks") or []
        if not joint_tasks:
            raise ValueError("joint_all config must define `joint_tasks`.")
    else:
        raise ValueError(f"Unsupported CL method: {method}")

    return config


def iter_step_names(config: Dict[str, Any]) -> Iterable[str]:
    if config["method"] == "joint_all":
        return ["all"]
    return config["steps"].keys()
