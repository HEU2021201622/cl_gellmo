import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Set, Tuple

from cl_data import normalize_task_name
from cl_eval_props import canonicalize_smiles

SMILES_PATTERN = re.compile(r"<SMILES>\s*([A-Za-z0-9@+\-\[\]\(\)\\/%=#$.:]+)\s*</SMILES>")


def load_json_records(path: Path) -> List[Dict]:
    with open(path, "r") as handle:
        return json.load(handle)


def load_train_manifest(path: Path) -> Dict:
    with open(path, "r") as handle:
        return json.load(handle)


def discover_sequence_jobs(config: Dict) -> List[Dict]:
    input_root = Path(config["input_root"])
    train_root = Path(config["train_root"])
    jobs = []
    steps_filter = config.get("_steps_filter")
    tasks_filter = config.get("_tasks_filter")
    declared_steps = set((config.get("steps") or {}).keys())
    step_dirs = sorted([path for path in input_root.iterdir() if path.is_dir()]) if input_root.exists() else []
    for step_dir in step_dirs:
        step_name = step_dir.name
        if steps_filter and step_name not in steps_filter:
            continue
        if declared_steps and step_name not in declared_steps:
            continue
        train_manifest = load_train_manifest(train_root / step_name / "train_manifest.json")
        for task_dir in sorted([path for path in step_dir.iterdir() if path.is_dir()]):
            task_name = normalize_task_name(task_dir.name)
            if tasks_filter and task_name not in tasks_filter:
                continue
            for setting_dir in sorted([path for path in task_dir.iterdir() if path.is_dir()]):
                setting = setting_dir.name
                if setting not in config["settings"]:
                    continue
                prediction_path = setting_dir / "predictions.json"
                if not prediction_path.exists():
                    continue
                jobs.append(
                    {
                        "mode": "sequence",
                        "unit_name": step_name,
                        "task": task_name,
                        "setting": setting,
                        "prediction_path": prediction_path,
                        "train_manifest": train_manifest,
                    }
                )
    return jobs


def discover_joint_jobs(config: Dict) -> List[Dict]:
    input_root = Path(config["input_root"])
    train_root = Path(config["train_root"])
    jobs = []
    runs_filter = config.get("_runs_filter")
    tasks_filter = config.get("_tasks_filter")
    declared_runs = set((config.get("runs") or {}).keys())
    run_dirs = sorted([path for path in input_root.iterdir() if path.is_dir()]) if input_root.exists() else []
    for run_dir in run_dirs:
        run_name = run_dir.name
        if runs_filter and run_name not in runs_filter:
            continue
        if declared_runs and run_name not in declared_runs:
            continue
        train_manifest = load_train_manifest(train_root / run_name / "train_manifest.json")
        for task_dir in sorted([path for path in run_dir.iterdir() if path.is_dir()]):
            task_name = normalize_task_name(task_dir.name)
            if tasks_filter and task_name not in tasks_filter:
                continue
            for setting_dir in sorted([path for path in task_dir.iterdir() if path.is_dir()]):
                setting = setting_dir.name
                if setting not in config["settings"]:
                    continue
                prediction_path = setting_dir / "predictions.json"
                if not prediction_path.exists():
                    continue
                jobs.append(
                    {
                        "mode": "joint",
                        "unit_name": run_name,
                        "task": task_name,
                        "setting": setting,
                        "prediction_path": prediction_path,
                        "train_manifest": train_manifest,
                    }
                )
    return jobs


def discover_jobs(config: Dict) -> List[Dict]:
    return discover_sequence_jobs(config) if config["mode"] == "sequence" else discover_joint_jobs(config)


def extract_candidate_smiles(response_list: Sequence[str]) -> List[str]:
    candidates = []
    for response in response_list or []:
        matches = SMILES_PATTERN.findall(response or "")
        for match in matches:
            canonical = canonicalize_smiles(match.strip())
            if canonical and canonical not in candidates:
                candidates.append(canonical)
    return candidates


def task_properties(task: str) -> List[str]:
    return normalize_task_name(task).split("+")


def collect_prediction_candidates(jobs: Iterable[Dict]) -> Tuple[Set[str], Set[str]]:
    unique_smiles: Set[str] = set()
    required_properties: Set[str] = set()
    for job in jobs:
        required_properties.update(task_properties(job["task"]))
        predictions = load_json_records(Path(job["prediction_path"]))
        for record in predictions:
            unique_smiles.update(extract_candidate_smiles(record.get("response", [])))
    return unique_smiles, required_properties
