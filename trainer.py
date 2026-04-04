import os
import sys
from pathlib import Path
from typing import List, Mapping, Optional, Sequence

import pandas as pd
import safetensors.torch
import torch
from datasets import Dataset, disable_caching, load_dataset
from peft import LoraConfig, get_peft_model, set_peft_model_state_dict
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
)
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR

from config import LORA_TARGET_MODULES_BY_FAMILY
from prompter import Prompter

disable_caching()
os.environ["TOKENIZERS_PARALLELISM"] = "false"


class SavePeftModelCallback(TrainerCallback):
    def on_save(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        checkpoint_folder = os.path.join(args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}")

        kwargs["model"].save_pretrained(checkpoint_folder)

        pytorch_model_path = os.path.join(checkpoint_folder, "pytorch_model.bin")
        torch.save({}, pytorch_model_path)
        return control


class LoadBestPeftModelCallback(TrainerCallback):
    def on_train_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        print(f"Loading best peft model from {state.best_model_checkpoint} (score: {state.best_metric}).")
        candidate_paths = [
            os.path.join(state.best_model_checkpoint, "adapter_model.safetensors"),
            os.path.join(state.best_model_checkpoint, "adapter_model.bin"),
        ]
        best_model_path = next((path for path in candidate_paths if os.path.exists(path)), None)
        if best_model_path is None:
            print(f"Best-model reload skipped because no adapter weights were found in {state.best_model_checkpoint}.")
            return control

        if best_model_path.endswith(".safetensors"):
            adapters_weights = safetensors.torch.load_file(best_model_path)
        else:
            adapters_weights = torch.load(best_model_path, weights_only=True)
        model = kwargs["model"]
        set_peft_model_state_dict(model, adapters_weights)
        return control


def _build_training_args(
    *,
    has_val: bool,
    gradient_checkpointing: bool,
    micro_batch_size: int,
    gradient_accumulation_steps: int,
    warmup_ratio: float,
    num_epochs: int,
    learning_rate: float,
    bf16: bool,
    optim: str,
    lr_scheduler: str,
    output_dir: str,
    ddp,
    group_by_length: bool,
    use_wandb: bool,
    wandb_run_name: str,
):
    args = dict(
        gradient_checkpointing=gradient_checkpointing,
        per_device_train_batch_size=micro_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        warmup_ratio=warmup_ratio,
        num_train_epochs=num_epochs,
        learning_rate=learning_rate,
        bf16=bf16,
        logging_steps=1,
        optim=optim,
        lr_scheduler_type=lr_scheduler,
        output_dir=output_dir,
        save_total_limit=2,
        ddp_find_unused_parameters=False if ddp else None,
        group_by_length=group_by_length,
        report_to="wandb" if use_wandb else None,
        run_name=wandb_run_name if use_wandb else None,
    )

    if has_val:
        args.update(
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
        )
    else:
        args.update(
            eval_strategy="no",
            save_strategy="steps",
            save_steps=1000,
            load_best_model_at_end=False,
        )
    return TrainingArguments(**args)


def _print_train_params(params: Mapping[str, object]) -> None:
    if int(os.environ.get("LOCAL_RANK", 0)) != 0:
        return
    lines = [f"{key}: {value}" for key, value in params.items()]
    print("\n".join(lines))


def _load_model_and_tokenizer(
    base_model: str,
    load_in_8bit: bool,
    device_map,
):
    quantization_config = BitsAndBytesConfig(load_in_8bit=load_in_8bit)
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,
        device_map=device_map,
        trust_remote_code=True,
        quantization_config=quantization_config,
    )

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    tokenizer.pad_token_id = 0
    tokenizer.padding_side = "right"

    bos = tokenizer.bos_token_id
    eos = tokenizer.eos_token_id
    pad = tokenizer.pad_token_id
    print("pre-trained model's BOS EOS and PAD token id:", bos, eos, pad, "=> pad is reset to 0")
    return model, tokenizer


def _infer_model_family(base_model: str, model_config=None) -> Optional[str]:
    candidates = []
    if model_config is not None:
        for attr in ("model_type", "architectures", "_name_or_path"):
            value = getattr(model_config, attr, None)
            if isinstance(value, str):
                candidates.append(value)
            elif isinstance(value, list):
                candidates.extend(str(item) for item in value)

    base_name = Path(base_model).name
    candidates.extend([base_model, base_name])
    haystack = " ".join(str(item).lower() for item in candidates if item)

    if "mistral" in haystack:
        return "mistral"
    if "llama" in haystack or "llama-3" in haystack or "llama3" in haystack:
        return "llama"
    return None


def resolve_lora_target_modules(base_model: str, model_config=None, lora_target_modules: Optional[List[str]] = None) -> List[str]:
    if lora_target_modules:
        return lora_target_modules

    family = _infer_model_family(base_model, model_config=model_config)
    if family and family in LORA_TARGET_MODULES_BY_FAMILY:
        resolved = list(LORA_TARGET_MODULES_BY_FAMILY[family])
        print(f"Auto-selected LoRA target modules for {family}: {resolved}")
        return resolved

    fallback = list(LORA_TARGET_MODULES_BY_FAMILY["mistral"])
    print(f"Falling back to default LoRA target modules: {fallback}")
    return fallback


def _build_prompt_tokenizer(
    tokenizer,
    prompter: Prompter,
    cutoff_len: int,
    train_on_inputs: bool,
    add_eos_token: bool,
    opt_type: str,
):
    def tokenize(prompt, add_eos=True):
        result = tokenizer(prompt, return_tensors=None, padding=False, truncation=True, max_length=cutoff_len)
        if (
            result["input_ids"][-1] != tokenizer.eos_token_id
            and len(result["input_ids"]) < cutoff_len
            and add_eos
        ):
            result["input_ids"].append(tokenizer.eos_token_id)
            result["attention_mask"].append(1)

        result["labels"] = result["input_ids"].copy()
        return result

    def generate_and_tokenize_prompt(data_point):
        full_prompt = prompter.generate_prompt(
            data_point,
            True,
            "ins",
            opt_type,
            add_response=True,
        )
        tokenized_full_prompt = tokenize(full_prompt)
        if not train_on_inputs:
            user_prompt = prompter.generate_prompt(
                data_point,
                True,
                "ins",
                opt_type,
                add_response=False,
            )
            tokenized_user_prompt = tokenize(user_prompt, add_eos=add_eos_token)
            user_prompt_len = len(tokenized_user_prompt["input_ids"])
            if add_eos_token:
                user_prompt_len -= 1

            tokenized_full_prompt["labels"] = [-100] * user_prompt_len + tokenized_full_prompt["labels"][user_prompt_len:]
        return tokenized_full_prompt

    return generate_and_tokenize_prompt


def _ensure_dataset(data: Optional[object]) -> Dataset:
    if data is None:
        return Dataset.from_pandas(pd.DataFrame())
    if isinstance(data, Dataset):
        return data
    if isinstance(data, pd.DataFrame):
        return Dataset.from_pandas(data.reset_index(drop=True))
    if isinstance(data, Sequence):
        return Dataset.from_pandas(pd.DataFrame(list(data)))
    raise TypeError(f"Unsupported dataset type: {type(data)}")


def train_on_records(
    *,
    base_model: str,
    train_data,
    val_data=None,
    output_dir: str,
    batch_size: int = 128,
    micro_batch_size: int = 1,
    num_epochs: int = 1,
    learning_rate: float = 1e-4,
    cutoff_len: int = 4096,
    lr_scheduler: str = "cosine",
    warmup_ratio: float = 0.1,
    lora_r: int = 16,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    lora_target_modules: List[str] = None,
    lora_weight_path: str = None,
    load_in_8bit: bool = False,
    train_on_inputs: bool = False,
    add_eos_token: bool = False,
    group_by_length: bool = False,
    gradient_checkpointing: bool = False,
    wandb_project: str = "",
    wandb_run_name: str = "",
    wandb_watch: str = "",
    wandb_log_model: str = "",
    resume_from_checkpoint: str = None,
    prompt_template_name: str = "mistral",
    opt_type: str = "simple",
    optim: str = "adamw_torch",
    log_prefix: str = "",
):
    assert base_model, "Please specify a --base_model"

    gradient_accumulation_steps = batch_size // micro_batch_size
    prompter = Prompter(opt_type)

    device_map = "auto"
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if ddp:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
        gradient_accumulation_steps = gradient_accumulation_steps // world_size
        print("gradient_accumulation_steps:", gradient_accumulation_steps)

    use_wandb = len(wandb_project) > 0 or ("WANDB_PROJECT" in os.environ and os.environ["WANDB_PROJECT"])
    if len(wandb_project) > 0:
        os.environ["WANDB_PROJECT"] = wandb_project
    if len(wandb_watch) > 0:
        os.environ["WANDB_WATCH"] = wandb_watch
    if len(wandb_log_model) > 0:
        os.environ["WANDB_LOG_MODEL"] = wandb_log_model

    train_dataset = _ensure_dataset(train_data)
    val_dataset = _ensure_dataset(val_data)

    model, tokenizer = _load_model_and_tokenizer(
        base_model=base_model,
        load_in_8bit=load_in_8bit,
        device_map=device_map,
    )
    lora_target_modules = resolve_lora_target_modules(
        base_model=base_model,
        model_config=model.config,
        lora_target_modules=lora_target_modules,
    )
    bf16 = True

    prompt_mapper = _build_prompt_tokenizer(
        tokenizer=tokenizer,
        prompter=prompter,
        cutoff_len=cutoff_len,
        train_on_inputs=train_on_inputs,
        add_eos_token=add_eos_token,
        opt_type=opt_type,
    )

    config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=lora_target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, config)

    if lora_weight_path:
        model.load_adapter(lora_weight_path, adapter_name="default")
    if resume_from_checkpoint:
        print(f"Resuming from checkpoint: {resume_from_checkpoint}")
        model.load_adapter(resume_from_checkpoint, adapter_name="default")

    model.print_trainable_parameters()

    train_dataset = train_dataset.shuffle().map(prompt_mapper)
    val_dataset = val_dataset.map(prompt_mapper) if len(val_dataset) else val_dataset

    if not ddp and torch.cuda.device_count() > 1:
        model.is_parallelizable = True
        model.model_parallel = True

    num_tokens = sum(len(row["input_ids"]) for row in train_dataset)
    print(f"{log_prefix}#tokens: {num_tokens / 1000.0:.1f}k")

    has_val = len(val_dataset) > 0
    callbacks = [SavePeftModelCallback()]
    if has_val:
        callbacks.append(LoadBestPeftModelCallback())

    trainer = Trainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=val_dataset if has_val else None,
        args=_build_training_args(
            has_val=has_val,
            gradient_checkpointing=gradient_checkpointing,
            micro_batch_size=micro_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_ratio=warmup_ratio,
            num_epochs=num_epochs,
            learning_rate=learning_rate,
            bf16=bf16,
            optim=optim,
            lr_scheduler=lr_scheduler,
            output_dir=output_dir,
            ddp=ddp,
            group_by_length=group_by_length,
            use_wandb=use_wandb,
            wandb_run_name=wandb_run_name,
        ),
        data_collator=DataCollatorForSeq2Seq(
            tokenizer,
            pad_to_multiple_of=8,
            return_tensors="pt",
            padding=True,
        ),
        callbacks=callbacks,
    )
    model.config.use_cache = False

    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    model.save_pretrained(output_dir)
    pytorch_model_path = os.path.join(output_dir, "pytorch_model.bin")
    torch.save({}, pytorch_model_path)
    tokenizer.save_pretrained(output_dir)


def _load_task_data(data_path: str, tasks: Sequence[str], train_set_size: Optional[int], val_set_size: int):
    data = load_dataset(data_path)["train"]
    data = pd.DataFrame(data)

    train_data = data[(data["split"] == "train") & (data["task"].isin(tasks))]
    val_data = data[(data["split"] == "val") & (data["task"].isin(tasks))]

    if train_set_size is not None:
        train_data = train_data.groupby("task").head(train_set_size).reset_index(drop=True)
    else:
        train_data = train_data.reset_index(drop=True)

    if val_set_size:
        val_data = val_data.groupby("task").head(val_set_size).reset_index(drop=True)
    else:
        val_data = val_data.iloc[0:0].reset_index(drop=True)
    return train_data, val_data


def train(
    base_model: str = "",
    data_path: str = "",
    tasks: List[str] = [],
    output_dir: str = "",
    batch_size: int = 128,
    micro_batch_size: int = 1,
    num_epochs: int = 1,
    learning_rate: float = 1e-4,
    cutoff_len: int = 4096,
    train_set_size: int = None,
    val_set_size: int = 0,
    lr_scheduler: str = "cosine",
    warmup_ratio: float = 0.1,
    lora_r: int = 16,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    lora_target_modules: List[str] = None,
    lora_weight_path: str = None,
    load_in_8bit: bool = False,
    train_on_inputs: bool = False,
    add_eos_token: bool = False,
    group_by_length: bool = False,
    gradient_checkpointing: bool = False,
    wandb_project: str = "",
    wandb_run_name: str = "",
    wandb_watch: str = "",
    wandb_log_model: str = "",
    resume_from_checkpoint: str = None,
    prompt_template_name: str = "mistral",
    opt_type: str = "simple",
    optim: str = "adamw_torch",
):
    _print_train_params(
        {
            "Params using prompt template": prompt_template_name,
            "tasks": tasks,
            "opt_type": opt_type,
            "base_model": base_model,
            "data_path": data_path,
            "output_dir": output_dir,
            "batch_size": batch_size,
            "micro_batch_size": micro_batch_size,
            "num_epochs": num_epochs,
            "learning_rate": learning_rate,
            "cutoff_len": cutoff_len,
            "train_set_size": train_set_size,
            "val_set_size": val_set_size,
            "lr_scheduler": lr_scheduler,
            "warmup_ratio": warmup_ratio,
            "lora_r": lora_r,
            "lora_alpha": lora_alpha,
            "lora_dropout": lora_dropout,
            "lora_target_modules": lora_target_modules,
            "lora_weight_path": lora_weight_path,
            "load_in_8bit": load_in_8bit,
            "train_on_inputs": train_on_inputs,
            "add_eos_token": add_eos_token,
            "group_by_length": group_by_length,
            "gradient_checkpointing": gradient_checkpointing,
            "optim": optim,
            "wandb_project": wandb_project,
            "wandb_run_name": wandb_run_name,
            "wandb_watch": wandb_watch,
            "wandb_log_model": wandb_log_model,
            "resume_from_checkpoint": resume_from_checkpoint or False,
        }
    )

    if not tasks:
        sys.exit("Please specify a --tasks")

    train_data, val_data = _load_task_data(
        data_path=data_path,
        tasks=tasks,
        train_set_size=train_set_size,
        val_set_size=val_set_size,
    )

    prompter = Prompter(opt_type)
    for task in tasks:
        if train_data[train_data["task"] == task].empty:
            continue
        print(f"Task: {task}")
        example_data = train_data[train_data["task"] == task].iloc[0]
        print(prompter.generate_prompt(example_data, True, "ins", opt_type, add_response=True))

    train_on_records(
        base_model=base_model,
        train_data=train_data,
        val_data=val_data,
        output_dir=output_dir,
        batch_size=batch_size,
        micro_batch_size=micro_batch_size,
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        cutoff_len=cutoff_len,
        lr_scheduler=lr_scheduler,
        warmup_ratio=warmup_ratio,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        lora_target_modules=lora_target_modules,
        lora_weight_path=lora_weight_path,
        load_in_8bit=load_in_8bit,
        train_on_inputs=train_on_inputs,
        add_eos_token=add_eos_token,
        group_by_length=group_by_length,
        gradient_checkpointing=gradient_checkpointing,
        wandb_project=wandb_project,
        wandb_run_name=wandb_run_name,
        wandb_watch=wandb_watch,
        wandb_log_model=wandb_log_model,
        resume_from_checkpoint=resume_from_checkpoint,
        prompt_template_name=prompt_template_name,
        opt_type=opt_type,
        optim=optim,
    )


if __name__ == "__main__":
    import fire

    torch.cuda.empty_cache()
    fire.Fire(train)
