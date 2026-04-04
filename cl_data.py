import csv
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import pandas as pd

from config import PROPERTY_ALIASES


FIELD_ALIASES = {
    "source_smiles": ["source_smiles", "source", "src_smiles", "input_smiles"],
    "target_smiles": ["target_smiles", "target", "tgt_smiles", "output_smiles"],
    "task": ["task", "task_label", "task_name"],
    "instruction": ["instruction", "prompt", "input_text"],
    "instr_idx": ["instr_idx", "instruction_idx"],
    "instr_setting": ["instr_setting", "instruction_setting"],
    "properties": ["properties", "property_scores", "props"],
    "scaffold": ["scaffold", "source_scaffold"],
    "similarity": ["similarity", "sim", "source_target_similarity", "pair_similarity"],
    "split": ["split", "dataset_split"],
}


def normalize_property_name(name: str) -> str:
    normalized = name.strip().lower().replace("-", "_")
    return PROPERTY_ALIASES.get(normalized, normalized)


def normalize_task_name(task_name: str) -> str:
    if not task_name:
        return task_name
    parts = [normalize_property_name(part) for part in task_name.split("+")]
    return "+".join(parts)


def _pick_value(record: Dict, field_name: str, default=None):
    for candidate in FIELD_ALIASES[field_name]:
        if candidate in record:
            return record[candidate]
    return default


def _normalize_properties(properties: Dict) -> Dict:
    normalized = {}
    if not isinstance(properties, dict):
        return normalized
    for key, value in properties.items():
        normalized[normalize_property_name(str(key))] = value
    return normalized


def normalize_record(record: Dict) -> Dict:
    normalized = {
        "source_smiles": _pick_value(record, "source_smiles", ""),
        "target_smiles": _pick_value(record, "target_smiles", ""),
        "task": normalize_task_name(_pick_value(record, "task", "")),
        "instruction": _pick_value(record, "instruction"),
        "instr_idx": _pick_value(record, "instr_idx", 0),
        "instr_setting": _pick_value(record, "instr_setting", "seen"),
        "properties": _normalize_properties(_pick_value(record, "properties", {})),
        "scaffold": _pick_value(record, "scaffold", ""),
        "similarity": _pick_value(record, "similarity"),
        "split": _pick_value(record, "split", "train"),
    }
    return normalized


def load_pairwise_records(path: str) -> List[Dict]:
    file_path = Path(path)
    suffix = file_path.suffix.lower()
    records: List[Dict] = []

    if suffix == ".jsonl":
        with open(file_path, "r", encoding="utf-8-sig") as handle:
            first_nonempty = None
            for raw_line in handle:
                stripped = raw_line.strip()
                if not stripped or stripped.startswith("#"):
                    continue
                first_nonempty = stripped
                break

        if first_nonempty is None:
            return records

        if first_nonempty.startswith("["):
            with open(file_path, "r", encoding="utf-8-sig") as handle:
                data = json.load(handle)
            if not isinstance(data, list):
                raise ValueError(f"Expected a JSON array in {file_path}, got {type(data)}")
            return [normalize_record(record) for record in data]

        with open(file_path, "r", encoding="utf-8-sig") as handle:
            for line_no, raw_line in enumerate(handle, start=1):
                line = raw_line.strip()
                if not line or line.startswith("#"):
                    continue
                try:
                    records.append(normalize_record(json.loads(line)))
                except json.JSONDecodeError as exc:
                    preview = line[:200]
                    raise ValueError(
                        f"Failed to parse JSONL record in {file_path} at line {line_no}: {preview}"
                    ) from exc
    elif suffix == ".json":
        with open(file_path, "r", encoding="utf-8-sig") as handle:
            data = json.load(handle)
        records = [normalize_record(record) for record in data]
    elif suffix == ".csv":
        with open(file_path, "r", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            records = [normalize_record(row) for row in reader]
    else:
        raise ValueError(f"Unsupported training data format: {file_path}")

    return records


def _sample_records(records: Sequence[Dict], limit: int, seed: int) -> List[Dict]:
    records = list(records)
    if limit is None or limit <= 0 or len(records) <= limit:
        return records
    rng = random.Random(seed)
    return rng.sample(records, limit)


def _sample_per_task(records: Sequence[Dict], tasks: Sequence[str], split: str, limit: int, seed: int) -> List[Dict]:
    sampled = []
    for index, task in enumerate(tasks):
        task_records = filter_records(records, split, [task])
        sampled.extend(_sample_records(task_records, limit, seed + index))
    return sampled


def _group_counts(records: Iterable[Dict], key: str) -> Dict[str, int]:
    counter = defaultdict(int)
    for record in records:
        counter[record.get(key) or ""] += 1
    return dict(counter)


def filter_records(records: Sequence[Dict], split: str, tasks: Sequence[str]) -> List[Dict]:
    normalized_tasks = {normalize_task_name(task) for task in tasks}
    return [
        record
        for record in records
        if record.get("split") == split and normalize_task_name(record.get("task", "")) in normalized_tasks
    ]


def build_step_dataset(
    records: Sequence[Dict],
    *,
    step_name: str,
    new_tasks: Sequence[str],
    combo_tasks: Sequence[str],
    new_limit: int,
    combo_limit: int,
    val_limit: int,
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
    train_new = _sample_records(filter_records(records, "train", new_tasks), new_limit, seed)
    train_combo = _sample_records(filter_records(records, "train", combo_tasks), combo_limit, seed + 1)
    val_new = _sample_records(filter_records(records, "val", new_tasks), val_limit, seed + 2)
    val_combo = _sample_records(filter_records(records, "val", combo_tasks), val_limit, seed + 3)

    train_records = train_new + train_combo
    val_records = val_new + val_combo
    metadata = {
        "step": step_name,
        "new_tasks": [normalize_task_name(task) for task in new_tasks],
        "combo_tasks": [normalize_task_name(task) for task in combo_tasks],
        "train_new_count": len(train_new),
        "train_combo_count": len(train_combo),
        "val_new_count": len(val_new),
        "val_combo_count": len(val_combo),
        "train_task_counts": _group_counts(train_records, "task"),
        "val_task_counts": _group_counts(val_records, "task"),
    }
    return pd.DataFrame(train_records), pd.DataFrame(val_records), metadata


def build_joint_dataset(
    records: Sequence[Dict],
    *,
    tasks: Sequence[str],
    train_limit: int,
    val_limit: int,
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
    per_task_train_limit = train_limit if train_limit and train_limit > 0 else None
    per_task_val_limit = val_limit if val_limit and val_limit > 0 else None
    train_records = _sample_per_task(records, tasks, "train", per_task_train_limit, seed)
    val_records = _sample_per_task(records, tasks, "val", per_task_val_limit, seed + 1000)
    metadata = {
        "step": "all",
        "joint_tasks": [normalize_task_name(task) for task in tasks],
        "train_count": len(train_records),
        "val_count": len(val_records),
        "train_task_counts": _group_counts(train_records, "task"),
        "val_task_counts": _group_counts(val_records, "task"),
    }
    return pd.DataFrame(train_records), pd.DataFrame(val_records), metadata
