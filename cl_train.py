import argparse
import random
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
import yaml

from cl_buffer import ReplayBuffer
from cl_config import iter_step_names, load_cl_config
from cl_data import build_joint_dataset, build_step_dataset, load_pairwise_records
from cl_manifest import ensure_dir, write_json, write_jsonl


def parse_args():
    parser = argparse.ArgumentParser(description="Continuous-learning training for GeLLMO.")
    parser.add_argument("--config", required=True, help="Path to configs/cl/*.yaml")
    parser.add_argument("--method", default=None, help="Override training method.")
    parser.add_argument("--output-root", default=None, help="Override output root.")
    parser.add_argument("--base-model", default=None, help="Override base model.")
    parser.add_argument("--data-root", default=None, help="Override data root.")
    parser.add_argument("--train-file", default=None, help="Override train file path relative to data root.")
    parser.add_argument("--seed", type=int, default=None, help="Override random seed.")
    parser.add_argument("--buffer-capacity", type=int, default=None, help="Override replay buffer capacity.")
    parser.add_argument("--replay-ratio", type=float, default=None, help="Override replay ratio.")
    parser.add_argument("--start-step", default=None, help="First step to run.")
    parser.add_argument("--end-step", default=None, help="Last step to run.")
    parser.add_argument("--resume-adapter", default=None, help="Explicit adapter path for the first executed step.")
    parser.add_argument("--dry-run", action="store_true", help="Prepare datasets and manifests without training.")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def apply_overrides(config: Dict, args) -> Dict:
    if args.method:
        config["method"] = args.method
    if args.base_model:
        config["base_model"] = args.base_model
    if args.data_root:
        config["data_root"] = args.data_root
    if args.train_file:
        config["train_file"] = args.train_file
    if args.seed is not None:
        config["seed"] = args.seed
    if args.buffer_capacity is not None:
        config["buffer_capacity"] = args.buffer_capacity
    if args.replay_ratio is not None:
        config["replay_ratio"] = args.replay_ratio
    if args.output_root:
        config["output_root"] = args.output_root
    return config


def slice_steps(step_names: List[str], start_step: str, end_step: str) -> List[str]:
    start_idx = 0
    end_idx = len(step_names)
    if start_step:
        start_idx = step_names.index(start_step)
    if end_step:
        end_idx = step_names.index(end_step) + 1
    return step_names[start_idx:end_idx]


def resolve_output_root(config: Dict) -> str:
    if config.get("output_root"):
        return config["output_root"]
    return f"outputs/cl_train/{config['method']}"


def save_resolved_config(step_dir: str, config: Dict) -> None:
    ensure_dir(step_dir)
    with open(Path(step_dir) / "train_config.resolved.yaml", "w") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)


def dataframe_records(frame: pd.DataFrame) -> List[Dict]:
    if frame.empty:
        return []
    return frame.to_dict(orient="records")


def train_step(
    *,
    step_name: str,
    step_dir: str,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    config: Dict,
    resume_adapter: str,
    manifest_extra: Dict,
    dry_run: bool,
) -> None:
    ensure_dir(step_dir)
    save_resolved_config(step_dir, config)

    if config.get("save_step_records", True) and not dry_run:
        write_jsonl(str(Path(step_dir) / "step_train_records.jsonl"), dataframe_records(train_df))
        write_jsonl(str(Path(step_dir) / "step_val_records.jsonl"), dataframe_records(val_df))

    manifest = {
        "method": config["method"],
        "step": step_name,
        "base_model": config["base_model"],
        "checkpoint_dir": step_dir,
        "resume_adapter": resume_adapter,
        "prompt_template_name": config["prompt_template_name"],
        "opt_type": config["opt_type"],
        "train_count": len(train_df),
        "val_count": len(val_df),
        **manifest_extra,
    }
    write_json(str(Path(step_dir) / "train_manifest.json"), manifest)

    if dry_run:
        print(f"[dry-run] prepared {step_name} at {step_dir}")
        return

    from trainer import train_on_records

    train_on_records(
        base_model=config["base_model"],
        train_data=train_df,
        val_data=val_df,
        output_dir=step_dir,
        batch_size=config["batch_size"],
        micro_batch_size=config["micro_batch_size"],
        num_epochs=config["num_epochs"],
        learning_rate=config["learning_rate"],
        cutoff_len=config["cutoff_len"],
        lr_scheduler=config["lr_scheduler"],
        warmup_ratio=config["warmup_ratio"],
        lora_target_modules=config["lora_target_modules"],
        load_in_8bit=config["load_in_8bit"],
        train_on_inputs=config["train_on_inputs"],
        add_eos_token=config["add_eos_token"],
        group_by_length=config["group_by_length"],
        gradient_checkpointing=config["gradient_checkpointing"],
        wandb_project=config["wandb_project"],
        wandb_run_name=config["wandb_run_name"] or f"{config['method']}-{step_name}",
        wandb_watch=config["wandb_watch"],
        wandb_log_model=config["wandb_log_model"],
        resume_from_checkpoint=resume_adapter,
        prompt_template_name=config["prompt_template_name"],
        opt_type=config["opt_type"],
        optim=config["optim"],
        log_prefix=f"[{step_name}] ",
    )


def main():
    args = parse_args()
    config = apply_overrides(load_cl_config(args.config), args)
    output_root = resolve_output_root(config)
    seed = config["seed"]
    set_seed(seed)

    data_path = str(Path(config["data_root"]) / config["train_file"])
    all_records = load_pairwise_records(data_path)
    ensure_dir(output_root)

    if config["method"] == "joint_all":
        step_dir = str(Path(output_root) / "all")
        train_df, val_df, metadata = build_joint_dataset(
            all_records,
            tasks=config["joint_tasks"],
            train_limit=config.get("train_limit", 0),
            val_limit=config["val_limit"],
            seed=seed,
        )
        train_step(
            step_name="all",
            step_dir=step_dir,
            train_df=train_df,
            val_df=val_df,
            config=config,
            resume_adapter=args.resume_adapter,
            manifest_extra=metadata,
            dry_run=args.dry_run,
        )
        return

    step_names = list(iter_step_names(config))
    selected_steps = slice_steps(step_names, args.start_step, args.end_step)
    replay_buffer = ReplayBuffer(
        capacity=config["buffer_capacity"],
        strategy=config["buffer_update_strategy"],
        seed=seed,
    )

    previous_adapter = args.resume_adapter
    for offset, step_name in enumerate(selected_steps):
        step_cfg = config["steps"][step_name]
        step_seed = seed + offset * 17
        base_train_df, val_df, metadata = build_step_dataset(
            all_records,
            step_name=step_name,
            new_tasks=step_cfg.get("new_tasks", []),
            combo_tasks=step_cfg.get("combo_tasks", []),
            new_limit=step_cfg.get("new_limit", config["new_limit"]),
            combo_limit=step_cfg.get("combo_limit", config["combo_limit"]),
            val_limit=step_cfg.get("val_limit", config["val_limit"]),
            seed=step_seed,
        )

        replay_records: List[Dict] = []
        if config["method"] == "replay":
            replay_records = replay_buffer.sample(
                current_data_size=len(base_train_df),
                replay_ratio=config["replay_ratio"],
            )

        replay_df = pd.DataFrame(replay_records)
        if replay_df.empty:
            train_df = base_train_df
        else:
            train_df = pd.concat([base_train_df, replay_df], ignore_index=True)

        step_dir = str(Path(output_root) / step_name)
        manifest_extra = {
            **metadata,
            "replay_count": len(replay_records),
            "buffer_capacity": config["buffer_capacity"],
            "replay_ratio": config["replay_ratio"],
            "buffer_strategy": config["buffer_update_strategy"],
        }
        train_step(
            step_name=step_name,
            step_dir=step_dir,
            train_df=train_df,
            val_df=val_df,
            config=config,
            resume_adapter=previous_adapter,
            manifest_extra=manifest_extra,
            dry_run=args.dry_run,
        )

        if config["method"] == "replay":
            replay_buffer.update(dataframe_records(base_train_df))
            if not args.dry_run:
                write_json(str(Path(step_dir) / "buffer_meta.json"), replay_buffer.metadata())
                replay_buffer.save_jsonl(str(Path(step_dir) / "buffer_snapshot.jsonl"))

        previous_adapter = step_dir


if __name__ == "__main__":
    main()
