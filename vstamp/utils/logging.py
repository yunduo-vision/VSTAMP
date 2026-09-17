from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Mapping


class CSVLogger:
    def __init__(self, path: str | Path, fieldnames: list[str]) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fieldnames = fieldnames
        if not self.path.exists():
            with self.path.open("w", newline="", encoding="utf-8") as stream:
                csv.DictWriter(stream, fieldnames=fieldnames).writeheader()

    def log(self, row: Mapping[str, Any]) -> None:
        with self.path.open("a", newline="", encoding="utf-8") as stream:
            csv.DictWriter(stream, fieldnames=self.fieldnames).writerow(
                {key: row.get(key, "") for key in self.fieldnames}
            )


def write_json(path: str | Path, payload: Any) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)

