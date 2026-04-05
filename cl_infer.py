import argparse
import gc
import json
import os
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from cl_config import load_cl_infer_config
from cl_data import filter_test_records, load_pairwise_records, normalize_task_name
from cl_manifest import ensure_dir, write_json, write_jsonl
from config import DEFAULT_MAX_NEW_TOKENS
from prompter import Prompter

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def parse_args():
    parser = argparse.ArgumentParser(description="Continuous-learning inference for GeLLMO.")
    parser.add_argument("--config", required=True, help="Path to configs/cl/infer_*.yaml")
    parser.add_argument("--mode", choices=["sequence", "joint"], default=None, help="Override infer mode.")
    parser.add_argument("--method", default=None, help="Override sequence method.")
    parser.add_argument("--joint-name", default=None, help="Override joint run name.")
    parser.add_argument("--base-model", default=None, help="Override base model.")
    parser.add_argument("--data-root", default=None, help="Override data root.")
    parser.add_argument("--test-file", default=None, help="Override test file path relative to data root.")
    parser.add_argument("--output-root", default=None, help="Override output root.")
    parser.add_argument("--settings", nargs="+", default=None, help="Override inference settings, e.g. seen unseen")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size.")
    parser.add_argument("--num-beams", type=int, default=None, help="Override number of beams.")
    parser.add_argument("--num-return-sequences", type=int, default=None, help="Override number of returned candidates.")
    parser.add_argument("--load-in-8bit", action="store_true", help="Load base model in 8bit.")
    parser.add_argument("--allow-non-500", action="store_true", help="Allow task/split slices whose size is not 500.")
    parser.add_argument("--dry-run", action="store_true", help="Resolve runs and write manifests without model generation.")
    return parser.parse_args()


def apply_overrides(config: Dict, args) -> Dict:
    config_output_root = config.get("output_root")
    if args.mode:
        config["mode"] = args.mode
    if args.method:
        config["method"] = args.method
    if args.joint_name:
        config["joint_name"] = args.joint_name
    if args.base_model:
        config["base_model"] = args.base_model
    if args.data_root:
        config["data_root"] = args.data_root
    if args.test_file:
        config["test_file"] = args.test_file
    if args.settings:
        config["settings"] = args.settings
    if args.batch_size is not None:
        config["batch_size"] = args.batch_size
    if args.num_beams is not None:
        config["num_beams"] = args.num_beams
    if args.num_return_sequences is not None:
        config["num_return_sequences"] = args.num_return_sequences
    if args.load_in_8bit:
        config["load_in_8bit"] = True
    if args.allow_non_500:
        config["allow_non_500"] = True
    if args.output_root:
        config["output_root"] = args.output_root

    if config["mode"] == "sequence" and args.output_root is None and (not config_output_root or args.method):
        config["output_root"] = f"outputs/cl_infer/sequence/{config['method']}"
    if config["mode"] == "joint" and args.output_root is None and (not config_output_root or args.joint_name):
        config["output_root"] = f"outputs/cl_infer/joint/{config['joint_name']}"
    return config


def load_manifest(checkpoint_dir: str) -> Dict:
    manifest_path = Path(checkpoint_dir) / "train_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing training manifest: {manifest_path}")
    with open(manifest_path, "r") as handle:
        return json.load(handle)


def derive_trained_tasks(manifest: Dict) -> List[str]:
    if manifest.get("trained_tasks"):
        return [normalize_task_name(task) for task in manifest["trained_tasks"]]
    if manifest.get("joint_tasks"):
        return [normalize_task_name(task) for task in manifest["joint_tasks"]]
    tasks = []
    for key in ("new_tasks", "combo_tasks"):
        tasks.extend(manifest.get(key, []))
    normalized = []
    for task in tasks:
        task_name = normalize_task_name(task)
        if task_name not in normalized:
            normalized.append(task_name)
    return normalized


def resolve_sequence_runs(config: Dict) -> List[Dict]:
    runs = []
    for step_name, step_cfg in config["steps"].items():
        checkpoint_dir = step_cfg.get("checkpoint_dir") or str(Path("outputs/cl_train") / config["method"] / step_name)
        manifest = load_manifest(checkpoint_dir)
        infer_tasks = step_cfg.get("infer_tasks") or derive_trained_tasks(manifest)
        runs.append(
            {
                "mode": "sequence",
                "run_name": step_name,
                "checkpoint_id": step_name,
                "checkpoint_dir": checkpoint_dir,
                "method_or_joint": config["method"],
                "base_model": config.get("base_model") or manifest["base_model"],
                "prompt_template_name": config.get("prompt_template_name") or manifest.get("prompt_template_name") or "mistral",
                "opt_type": config.get("opt_type") or manifest.get("opt_type") or "simple",
                "infer_tasks": [normalize_task_name(task) for task in infer_tasks],
            }
        )
    return runs


def resolve_joint_runs(config: Dict) -> List[Dict]:
    runs = []
    for run_name, run_cfg in config["runs"].items():
        checkpoint_dir = run_cfg.get("checkpoint_dir") or str(Path("outputs/cl_train") / config["joint_name"] / run_name)
        manifest = load_manifest(checkpoint_dir)
        infer_tasks = run_cfg.get("infer_tasks") or derive_trained_tasks(manifest)
        runs.append(
            {
                "mode": "joint",
                "run_name": run_name,
                "checkpoint_id": run_name,
                "checkpoint_dir": checkpoint_dir,
                "method_or_joint": config["joint_name"],
                "base_model": config.get("base_model") or manifest["base_model"],
                "prompt_template_name": config.get("prompt_template_name") or manifest.get("prompt_template_name") or "mistral",
                "opt_type": config.get("opt_type") or manifest.get("opt_type") or "simple",
                "infer_tasks": [normalize_task_name(task) for task in infer_tasks],
            }
        )
    return runs


def build_output_dir(config: Dict, run: Dict, task: str, setting: str) -> Path:
    if run["mode"] == "sequence":
        return Path(config["output_root"]) / run["run_name"] / task / setting
    return Path(config["output_root"]) / run["run_name"] / task / setting


def validate_records(records: List[Dict], *, task: str, setting: str, allow_non_500: bool) -> None:
    if not records:
        raise ValueError(f"No test records found for task={task}, setting={setting}")
    if not allow_non_500 and len(records) != 500:
        raise ValueError(
            f"Expected exactly 500 test records for task={task}, setting={setting}, found {len(records)}"
        )


def build_prompts(records: Iterable[Dict], *, prompter: Prompter, opt_type: str, setting: str) -> List[str]:
    prompts = []
    for record in records:
        prompts.append(
            prompter.generate_prompt(
                record,
                sample_is_pair=False,
                prompt_type="ins",
                opt_type=opt_type,
                add_response=False,
                instr_setting=setting,
            )
        )
    return prompts


def load_generation_stack(base_model: str, adapter_dir: str, *, load_in_8bit: bool):
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline

    quantization_config = BitsAndBytesConfig(load_in_8bit=load_in_8bit)
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,
        quantization_config=quantization_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(
        model,
        adapter_dir,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    if not load_in_8bit:
        model.bfloat16()

    model.eval()
    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

    eos_token_id = model.config.eos_token_id or tokenizer.eos_token_id
    if isinstance(eos_token_id, list):
        eos_token_id = eos_token_id[0]

    if not model.config.eos_token_id:
        tokenizer.pad_token_id = eos_token_id
        tokenizer.padding_side = "left"
        model.config.eos_token_id = eos_token_id
    else:
        tokenizer.pad_token_id = eos_token_id
        tokenizer.padding_side = "left"
    model.config.pad_token_id = tokenizer.pad_token_id

    generator = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    # `pipeline` batching requires a concrete pad token id. Some Llama configs expose
    # eos_token_id as a list, so normalize it and set it explicitly on the pipeline tokenizer.
    generator.tokenizer.pad_token_id = tokenizer.pad_token_id
    generator.model.config.pad_token_id = tokenizer.pad_token_id
    if hasattr(generator.model, "generation_config"):
        generator.model.generation_config.top_p = None
    return generator, tokenizer


def release_generation_stack(generator, tokenizer) -> None:
    if generator is not None:
        model = getattr(generator, "model", None)
        if model is not None:
            del model
        del generator
    if tokenizer is not None:
        del tokenizer

    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            if hasattr(torch.cuda, "ipc_collect"):
                torch.cuda.ipc_collect()
    except Exception:
        pass
    gc.collect()


def generate_predictions(
    *,
    prompts: List[str],
    prompter: Prompter,
    generator,
    tokenizer,
    batch_size: int,
    num_beams: int,
    num_return_sequences: int,
) -> List[List[str]]:
    outputs = generator(
        prompts,
        do_sample=False,
        max_new_tokens=DEFAULT_MAX_NEW_TOKENS,
        temperature=1.0,
        num_return_sequences=num_return_sequences,
        num_beams=num_beams,
        pad_token_id=tokenizer.eos_token_id,
        batch_size=batch_size,
    )
    results = []
    for item in outputs:
        if isinstance(item, dict):
            item = [item]
        results.append([prompter.get_response(candidate["generated_text"]) for candidate in item])
    return results


def build_prediction_records(
    *,
    records: List[Dict],
    prompts: List[str],
    responses: List[List[str]],
    task: str,
    setting: str,
    run: Dict,
) -> List[Dict]:
    payload = []
    for index, (record, prompt, response_list) in enumerate(zip(records, prompts, responses)):
        item = {
            "prompt": prompt,
            "response": response_list,
            "task": task,
            "split_type": setting,
            "checkpoint_id": run["checkpoint_id"],
            "checkpoint_dir": run["checkpoint_dir"],
            "base_model": run["base_model"],
            "source_smiles": record["source_smiles"],
            "record_index": index,
        }
        if record.get("source_id") is not None:
            item["source_id"] = record["source_id"]
        if record.get("instr_idx") is not None:
            item["instr_idx"] = record["instr_idx"]
        payload.append(item)
    return payload


def write_prediction_outputs(output_dir: Path, payload: List[Dict], manifest: Dict) -> None:
    ensure_dir(str(output_dir))
    write_json(str(output_dir / "predictions.json"), payload)
    write_jsonl(str(output_dir / "predictions.jsonl"), payload)
    write_json(str(output_dir / "infer_manifest.json"), manifest)


def main():
    args = parse_args()
    config = apply_overrides(load_cl_infer_config(args.config), args)
    if config["num_return_sequences"] > config["num_beams"]:
        raise ValueError("num_return_sequences must be <= num_beams for beam search inference.")

    data_path = str(Path(config["data_root"]) / config["test_file"])
    all_records = load_pairwise_records(data_path)
    response_split = "%%% Response:"

    runs = resolve_sequence_runs(config) if config["mode"] == "sequence" else resolve_joint_runs(config)

    for run in runs:
        prompter = Prompter(run["opt_type"], response_split, template_path="templates")
        generator = None
        tokenizer = None
        try:
            if not args.dry_run:
                generator, tokenizer = load_generation_stack(
                    run["base_model"],
                    run["checkpoint_dir"],
                    load_in_8bit=config["load_in_8bit"],
                )

            for setting in config["settings"]:
                for task in run["infer_tasks"]:
                    task_records = filter_test_records(all_records, tasks=[task], instr_setting=setting)
                    validate_records(
                        task_records,
                        task=task,
                        setting=setting,
                        allow_non_500=config["allow_non_500"],
                    )
                    prompts = build_prompts(task_records, prompter=prompter, opt_type=run["opt_type"], setting=setting)
                    responses = [[] for _ in task_records]
                    if not args.dry_run:
                        responses = generate_predictions(
                            prompts=prompts,
                            prompter=prompter,
                            generator=generator,
                            tokenizer=tokenizer,
                            batch_size=config["batch_size"],
                            num_beams=config["num_beams"],
                            num_return_sequences=config["num_return_sequences"],
                        )

                    payload = build_prediction_records(
                        records=task_records,
                        prompts=prompts,
                        responses=responses,
                        task=task,
                        setting=setting,
                        run=run,
                    )
                    output_dir = build_output_dir(config, run, task, setting)
                    manifest = {
                        "mode": run["mode"],
                        "method_or_joint": run["method_or_joint"],
                        "step_or_all": run["run_name"],
                        "task": task,
                        "split_type": setting,
                        "base_model": run["base_model"],
                        "adapter_dir": run["checkpoint_dir"],
                        "num_beams": config["num_beams"],
                        "num_return_sequences": config["num_return_sequences"],
                        "record_count": len(payload),
                        "dry_run": bool(args.dry_run),
                    }
                    write_prediction_outputs(output_dir, payload, manifest)
                    print(f"[{'dry-run' if args.dry_run else 'infer'}] wrote {len(payload)} records to {output_dir}")
        finally:
            release_generation_stack(generator, tokenizer)


if __name__ == "__main__":
    main()
