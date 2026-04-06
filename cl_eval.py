import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import pandas as pd

from cl_config import load_cl_eval_config
from cl_data import load_pairwise_records, normalize_task_name
from cl_eval_data import (
    discover_jobs,
    extract_candidate_smiles,
    load_json_records,
    task_properties,
)
from cl_eval_props import (
    composite_score,
    compute_fp_diversity,
    compute_scaffold_diversity,
    hypervolume,
    merge_property_caches,
    normalized_improvement,
    pair_similarity,
    safe_mean,
)
from cl_manifest import ensure_dir, write_json, write_jsonl
from config import PROPERTY_IMPV_THRESHOLDS
STEP_INTRODUCED_PROPERTY = {
    "step1": "drd2",
    "step2": "qed",
    "step3": "plogp",
    "step4": "jnk3",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Continuous-learning evaluation for GeLLMO.")
    parser.add_argument("--config", required=True, help="Path to configs/cl/eval_*.yaml")
    parser.add_argument("--mode", choices=["sequence", "joint"], default=None)
    parser.add_argument("--method", default=None)
    parser.add_argument("--joint-name", default=None)
    parser.add_argument("--input-root", default=None)
    parser.add_argument("--train-root", default=None)
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--test-file", default=None)
    parser.add_argument("--settings", nargs="+", default=None)
    parser.add_argument("--steps", nargs="+", default=None)
    parser.add_argument("--runs", nargs="+", default=None)
    parser.add_argument("--tasks", nargs="+", default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--local-cache-path", default=None)
    parser.add_argument("--tdc-cache-path", default=None)
    return parser.parse_args()


def apply_overrides(config: Dict, args) -> Dict:
    local_cache_overridden = args.local_cache_path is not None
    tdc_cache_overridden = args.tdc_cache_path is not None
    if args.mode:
        config["mode"] = args.mode
    if args.method:
        config["method"] = args.method
        if args.input_root is None:
            config["input_root"] = f"outputs/cl_infer/sequence/{config['method']}"
        if args.train_root is None:
            config["train_root"] = f"outputs/cl_train/{config['method']}"
        if args.output_root is None:
            config["output_root"] = f"outputs/cl_eval/sequence/{config['method']}"
    if args.joint_name:
        config["joint_name"] = args.joint_name
        if args.input_root is None:
            config["input_root"] = f"outputs/cl_infer/joint/{config['joint_name']}"
        if args.train_root is None:
            config["train_root"] = f"outputs/cl_train/{config['joint_name']}"
        if args.output_root is None:
            config["output_root"] = f"outputs/cl_eval/joint/{config['joint_name']}"
    if args.input_root:
        config["input_root"] = args.input_root
    if args.train_root:
        config["train_root"] = args.train_root
    if args.output_root:
        config["output_root"] = args.output_root
    if args.data_root:
        config["data_root"] = args.data_root
    if args.test_file:
        config["test_file"] = args.test_file
    if args.settings:
        config["settings"] = args.settings
    if args.steps:
        config["_steps_filter"] = set(args.steps)
    if args.runs:
        config["_runs_filter"] = set(args.runs)
    if args.tasks:
        config["_tasks_filter"] = {normalize_task_name(task) for task in args.tasks}
    if args.num_workers is not None:
        config["num_workers"] = args.num_workers
    if args.batch_size is not None:
        config["batch_size"] = args.batch_size
    if args.local_cache_path:
        config["local_cache_path"] = args.local_cache_path
    if args.tdc_cache_path:
        config["tdc_cache_path"] = args.tdc_cache_path
    if args.output_root and not local_cache_overridden:
        config["local_cache_path"] = str(Path(config["output_root"]) / "property_cache_local.csv")
    if args.output_root and not tdc_cache_overridden:
        config["tdc_cache_path"] = str(Path(config["output_root"]) / "property_cache_tdc.csv")
    return config


def load_seen_smiles(path: Path) -> set:
    with open(path, "r") as handle:
        return {line.strip() for line in handle if line.strip()}


def build_test_lookup(test_records: Iterable[Dict]) -> Dict:
    lookup = {}
    grouped = defaultdict(list)
    for record in test_records:
        grouped[(normalize_task_name(record["task"]), record.get("instr_setting", "seen"))].append(record)
    for key, records in grouped.items():
        for index, record in enumerate(records):
            lookup[(key[0], key[1], record["source_smiles"], index)] = record
    return lookup


def property_threshold(prop: str) -> float:
    return PROPERTY_IMPV_THRESHOLDS[prop]


def candidate_improvement(source_value: float, candidate_value: float, prop: str) -> float:
    return candidate_value - source_value


def percentage_improvement(source_value: float, candidate_value: float) -> float:
    if source_value == 0:
        return (candidate_value - 0.1) / 0.1
    return abs(candidate_value - source_value) / abs(source_value)


def choose_best_candidate(
    source_props: Dict[str, float],
    candidate_values: Dict[str, Dict[str, float]],
    task_props: Sequence[str],
    *,
    use_thresholds: bool,
):
    best = None
    best_score = -float("inf")
    for smiles, prop_values in candidate_values.items():
        improvements = []
        success = True
        for prop in task_props:
            source_value = source_props[prop]
            candidate_value = prop_values[prop]
            delta = candidate_improvement(source_value, candidate_value, prop)
            if delta <= 0:
                success = False
                break
            if use_thresholds and delta < property_threshold(prop):
                success = False
                break
            improvements.append(percentage_improvement(source_value, candidate_value))
        if not success:
            continue
        score = composite_score(improvements)
        if score >= best_score:
            best_score = score
            best = (smiles, prop_values, improvements)
    return best


def build_candidate_occurrences(predictions: List[Dict], test_lookup: Dict, task: str, setting: str):
    rows = []
    unique_smiles = set()
    for index, record in enumerate(predictions):
        source_smiles = record["source_smiles"]
        test_record = test_lookup.get((task, setting, source_smiles, index))
        if test_record is None:
            matches = [candidate for candidate in test_lookup.values() if candidate["task"] == task and candidate.get("instr_setting", "seen") == setting and candidate["source_smiles"] == source_smiles]
            test_record = matches[0] if matches else None
        candidates = extract_candidate_smiles(record.get("response", []))
        valid_candidates = []
        for smiles in candidates:
            unique_smiles.add(smiles)
            valid_candidates.append(smiles)
        rows.append(
            {
                "record": record,
                "test_record": test_record,
                "candidates": valid_candidates,
                "raw_candidate_count": len(record.get("response", [])),
            }
        )
    return rows, unique_smiles


def evaluate_task(job: Dict, test_lookup: Dict, seen_smiles: set, config: Dict) -> Dict:
    predictions = load_json_records(job["prediction_path"])
    task_props = task_properties(job["task"])
    candidate_rows, unique_smiles = build_candidate_occurrences(predictions, test_lookup, job["task"], job["setting"])
    property_df = merge_property_caches([config.get("local_cache_path"), config.get("tdc_cache_path")])
    property_df = property_df.drop_duplicates(subset=["smiles"], keep="last")
    property_df = property_df.set_index("smiles") if not property_df.empty else pd.DataFrame(columns=["smiles"]).set_index(pd.Index([]))
    for column in list(task_props) + ["sas"]:
        if column not in property_df.columns:
            property_df[column] = pd.Series(dtype=float)
    missing_columns = [column for column in task_props if column not in property_df.columns]
    if missing_columns and not property_df.empty:
        raise ValueError(
            f"Missing property columns {missing_columns} for task={job['task']}. "
            f"Run local/TDC property prediction first."
        )

    num_inputs = len(candidate_rows)
    num_valid_inputs = 0
    successful_smiles = []
    successful_input_sims = []
    successful_ri = []
    successful_sas = []
    successful_unseen = 0
    successful_smiles_dir = []
    successful_input_sims_dir = []
    successful_ri_dir = []
    unique_valid_occurrences = 0
    total_valid_occurrences = 0
    candidate_vectors = []
    best_candidate_records = []

    for row in candidate_rows:
        test_record = row["test_record"]
        if test_record is None:
            continue
        source_smiles = test_record["source_smiles"]
        source_props = {prop: test_record["properties"][prop]["source"] for prop in task_props}
        candidate_values = {}
        for candidate in row["candidates"]:
            if candidate not in property_df.index:
                continue
            prop_values = {prop: float(property_df.loc[candidate, prop]) for prop in task_props}
            candidate_values[candidate] = prop_values

            total_valid_occurrences += 1
            improvements = []
            hv_vector = []
            for prop in task_props:
                delta = candidate_improvement(source_props[prop], prop_values[prop], prop)
                score = normalized_improvement(delta, property_threshold(prop))
                improvements.append(score)
                hv_vector.append(max(0.0, score))
            candidate_vectors.append(
                {
                    "smiles": candidate,
                    "source_smiles": source_smiles,
                    "score": composite_score(improvements),
                    "vector": hv_vector,
                }
            )
        if candidate_values:
            num_valid_inputs += 1
            unique_valid_occurrences += len(set(candidate_values.keys()))

        best_candidate_dir = choose_best_candidate(source_props, candidate_values, task_props, use_thresholds=False)
        if best_candidate_dir is not None:
            smiles_dir, _, improvements_dir = best_candidate_dir
            successful_smiles_dir.append(smiles_dir)
            successful_input_sims_dir.append(pair_similarity(source_smiles, smiles_dir))
            successful_ri_dir.append(composite_score(improvements_dir))

        best_candidate = choose_best_candidate(source_props, candidate_values, task_props, use_thresholds=True)
        if best_candidate is None:
            continue

        smiles, prop_values, improvements = best_candidate
        successful_smiles.append(smiles)
        successful_input_sims.append(pair_similarity(source_smiles, smiles))
        successful_ri.append(composite_score(improvements))
        successful_sas.append(float(property_df.loc[smiles, "sas"]) if "sas" in property_df.columns else float("nan"))
        if smiles not in seen_smiles:
            successful_unseen += 1
        best_candidate_records.append(
            {
                "task": job["task"],
                "setting": job["setting"],
                "unit_name": job["unit_name"],
                "source_smiles": source_smiles,
                "optimized_smiles": smiles,
                "source_properties": source_props,
                "optimized_properties": prop_values,
            }
        )

    candidate_vectors = sorted(candidate_vectors, key=lambda item: item["score"], reverse=True)
    top10 = safe_mean([item["score"] for item in candidate_vectors[:10]])
    top100 = safe_mean([item["score"] for item in candidate_vectors[:100]])
    hv = hypervolume([item["vector"] for item in candidate_vectors])

    sr = (len(successful_smiles) / num_inputs) * 100 if num_inputs else 0.0
    validity = (num_valid_inputs / num_inputs) * 100 if num_inputs else 0.0
    novelty = (successful_unseen / len(successful_smiles)) * 100 if successful_smiles else 0.0
    uniqueness = (unique_valid_occurrences / total_valid_occurrences) * 100 if total_valid_occurrences else 0.0

    metrics = {
        "mode": job["mode"],
        "unit_name": job["unit_name"],
        "task": job["task"],
        "split_type": job["setting"],
        "task_order": len(task_props),
        "SR": sr,
        "SR_dir": (len(successful_smiles_dir) / num_inputs) * 100 if num_inputs else 0.0,
        "Sim": safe_mean(successful_input_sims),
        "Sim_dir": safe_mean(successful_input_sims_dir),
        "RI": safe_mean(successful_ri),
        "RI_dir": safe_mean(successful_ri_dir),
        "Validity": validity,
        "Novelty": novelty,
        "Uniqueness": uniqueness,
        "Scaffold diversity": compute_scaffold_diversity(successful_smiles),
        "Div": compute_fp_diversity(successful_smiles),
        "SAS": safe_mean(successful_sas),
        "top10": top10,
        "top100": top100,
        "HV": hv,
        "num_inputs": num_inputs,
        "num_success": len(successful_smiles),
        "num_valid_inputs": num_valid_inputs,
        "checkpoint_dir": job["train_manifest"]["checkpoint_dir"],
    }
    return {"metrics": metrics, "best_candidates": best_candidate_records}


def summarize_by_order(task_metrics_df: pd.DataFrame) -> pd.DataFrame:
    seen_df = task_metrics_df[task_metrics_df["split_type"] == "seen"].copy()
    if seen_df.empty:
        return pd.DataFrame(columns=["unit_name", "task_order", "SR", "Sim", "RI"])
    grouped = seen_df.groupby(["unit_name", "task_order"])[["SR", "Sim", "RI"]].mean().reset_index()
    grouped["task_order"] = grouped["task_order"].map({1: "single", 2: "double", 3: "triple", 4: "quadruple"})
    return grouped


def summarize_seen_unseen_gap(task_metrics_df: pd.DataFrame) -> pd.DataFrame:
    grouped = task_metrics_df.pivot_table(
        index=["mode", "unit_name", "task"],
        columns="split_type",
        values="SR",
        aggfunc="mean",
    ).reset_index()
    if "seen" not in grouped.columns:
        grouped["seen"] = math.nan
    if "unseen" not in grouped.columns:
        grouped["unseen"] = math.nan
    grouped["seen_unseen_gap"] = grouped["seen"] - grouped["unseen"]
    return grouped


def summarize_cl_sequence(task_metrics_df: pd.DataFrame) -> pd.DataFrame:
    seen_df = task_metrics_df[task_metrics_df["split_type"] == "seen"].copy()
    if seen_df.empty:
        return pd.DataFrame(columns=["unit_name", "Old-task average", "New-task average", "All-task average", "Average Seen-Task SR", "Forgetting", "Backward Transfer"])

    step_names = sorted(seen_df["unit_name"].unique())
    first_seen = {}
    previous_scores = {}
    rows = []
    for step_name in step_names:
        step_df = seen_df[seen_df["unit_name"] == step_name].copy()
        new_prop = STEP_INTRODUCED_PROPERTY.get(step_name)
        step_df["is_new_task"] = step_df["task"].str.split("+").apply(lambda props: new_prop in props if new_prop else False)
        old_df = step_df[~step_df["is_new_task"]]
        new_df = step_df[step_df["is_new_task"]]

        forgetting_values = []
        bwt_values = []
        for _, record in step_df.iterrows():
            task = record["task"]
            current_sr = record["SR"]
            if task in previous_scores:
                forgetting_values.append(previous_scores[task] - current_sr)
            if task in first_seen:
                bwt_values.append(current_sr - first_seen[task])
            else:
                first_seen[task] = current_sr
            previous_scores[task] = current_sr

        rows.append(
            {
                "unit_name": step_name,
                "Old-task average": old_df["SR"].mean() if not old_df.empty else math.nan,
                "New-task average": new_df["SR"].mean() if not new_df.empty else math.nan,
                "All-task average": step_df["SR"].mean() if not step_df.empty else math.nan,
                "Average Seen-Task SR": step_df["SR"].mean() if not step_df.empty else math.nan,
                "Forgetting": safe_mean(forgetting_values),
                "Backward Transfer": safe_mean(bwt_values),
            }
        )
    return pd.DataFrame(rows)


def summarize_cl_joint(task_metrics_df: pd.DataFrame) -> pd.DataFrame:
    seen_df = task_metrics_df[task_metrics_df["split_type"] == "seen"].copy()
    rows = []
    for run_name, run_df in seen_df.groupby("unit_name"):
        rows.append(
            {
                "unit_name": run_name,
                "Old-task average": math.nan,
                "New-task average": math.nan,
                "All-task average": run_df["SR"].mean() if not run_df.empty else math.nan,
                "Average Seen-Task SR": run_df["SR"].mean() if not run_df.empty else math.nan,
                "Forgetting": math.nan,
                "Backward Transfer": math.nan,
            }
        )
    return pd.DataFrame(rows)


def main():
    args = parse_args()
    config = apply_overrides(load_cl_eval_config(args.config), args)

    data_root = Path(config["data_root"])
    test_records = load_pairwise_records(str(data_root / config["test_file"]))
    seen_smiles = load_seen_smiles(data_root / config["seen_smiles_file"])
    test_lookup = build_test_lookup(test_records)

    jobs = discover_jobs(config)
    if not jobs:
        raise ValueError(f"No evaluation jobs discovered under {config['input_root']}")

    task_metric_rows = []
    best_candidate_rows = []
    for job in jobs:
        result = evaluate_task(job, test_lookup, seen_smiles, config)
        task_metric_rows.append(result["metrics"])
        best_candidate_rows.extend(result["best_candidates"])
        print(f"[eval] {job['mode']} {job['unit_name']} {job['task']} {job['setting']}")

    task_metrics_df = pd.DataFrame(task_metric_rows)
    order_summary_df = summarize_by_order(task_metrics_df)
    cl_summary_df = summarize_cl_sequence(task_metrics_df) if config["mode"] == "sequence" else summarize_cl_joint(task_metrics_df)
    gap_df = summarize_seen_unseen_gap(task_metrics_df)

    output_root = Path(config["output_root"])
    ensure_dir(str(output_root))
    task_metrics_df.to_csv(output_root / "task_level_metrics.csv", index=False)
    order_summary_df.to_csv(output_root / "metrics_summary.csv", index=False)
    cl_summary_df.to_csv(output_root / "cl_summary.csv", index=False)
    gap_df.to_csv(output_root / "seen_unseen_gap.csv", index=False)
    write_jsonl(str(output_root / "best_candidates.jsonl"), best_candidate_rows)
    write_json(
        str(output_root / "eval_manifest.json"),
        {
            "mode": config["mode"],
            "input_root": config["input_root"],
            "output_root": config["output_root"],
            "num_jobs": len(jobs),
            "settings": config["settings"],
            "num_workers": config["num_workers"],
            "batch_size": config["batch_size"],
        },
    )


if __name__ == "__main__":
    main()
