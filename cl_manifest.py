import json
from pathlib import Path
from typing import Dict, Iterable


def ensure_dir(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def write_json(path: str, payload: Dict) -> None:
    ensure_dir(str(Path(path).parent))
    with open(path, "w") as handle:
        json.dump(payload, handle, indent=2)


def write_jsonl(path: str, records: Iterable[Dict]) -> None:
    ensure_dir(str(Path(path).parent))
    with open(path, "w") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")
