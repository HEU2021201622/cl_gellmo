import json
import random
from collections import defaultdict
from typing import Dict, Iterable, List, Sequence


class ReplayBuffer:
    def __init__(self, capacity: int, strategy: str = "task-balanced", seed: int = 42):
        self.capacity = max(0, int(capacity))
        self.strategy = strategy
        self.seed = seed
        self.records: List[Dict] = []

    def __len__(self) -> int:
        return len(self.records)

    def _group_key(self, record: Dict) -> str:
        if self.strategy == "scaffold-balanced":
            scaffold = record.get("scaffold")
            if scaffold:
                return scaffold
        return record.get("task", "")

    def _balanced_trim(self, records: Sequence[Dict], limit: int) -> List[Dict]:
        if limit <= 0:
            return []
        grouped = defaultdict(list)
        for record in records:
            grouped[self._group_key(record)].append(record)

        rng = random.Random(self.seed + len(records))
        for items in grouped.values():
            rng.shuffle(items)

        kept: List[Dict] = []
        round_idx = 0
        keys = sorted(grouped.keys())
        while len(kept) < limit and keys:
            next_keys = []
            for key in keys:
                items = grouped[key]
                if round_idx < len(items) and len(kept) < limit:
                    kept.append(items[round_idx])
                if round_idx + 1 < len(items):
                    next_keys.append(key)
            keys = next_keys
            round_idx += 1
        return kept

    def update(self, new_records: Iterable[Dict]) -> None:
        merged = list(self.records) + list(new_records)
        self.records = self._balanced_trim(merged, self.capacity)

    def sample(self, current_data_size: int, replay_ratio: float) -> List[Dict]:
        if replay_ratio <= 0 or not self.records:
            return []
        target = min(len(self.records), int(round(current_data_size * replay_ratio)))
        return self._balanced_trim(self.records, target)

    def save_jsonl(self, path: str) -> None:
        with open(path, "w") as handle:
            for record in self.records:
                handle.write(json.dumps(record) + "\n")

    def metadata(self) -> Dict:
        task_counts = defaultdict(int)
        scaffold_counts = defaultdict(int)
        for record in self.records:
            task_counts[record.get("task", "")] += 1
            scaffold_counts[record.get("scaffold", "")] += 1
        return {
            "capacity": self.capacity,
            "strategy": self.strategy,
            "size": len(self.records),
            "task_counts": dict(task_counts),
            "scaffold_counts": dict(scaffold_counts),
        }
