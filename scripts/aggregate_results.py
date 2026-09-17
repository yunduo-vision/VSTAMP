from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Aggregate scalar metrics across matched seed directories")
    parser.add_argument("root", help="Experiment directory containing seed_*/metrics.json")
    parser.add_argument("--output", default="aggregate_metrics.json")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    root = Path(args.root)
    payloads = []
    for path in sorted(root.glob("seed_*/metrics.json")):
        payloads.append(json.loads(path.read_text(encoding="utf-8")))
    if not payloads:
        raise FileNotFoundError(f"No seed_*/metrics.json files below {root}")
    keys = sorted(set.intersection(*(set(item) for item in payloads)))
    result = {"seeds": len(payloads), "metrics": {}}
    for key in keys:
        values = [item[key] for item in payloads]
        if all(isinstance(value, (int, float)) for value in values):
            array = np.asarray(values, dtype=np.float64)
            result["metrics"][key] = {
                "mean": float(array.mean()),
                "sample_std": float(array.std(ddof=1)) if len(array) > 1 else 0.0,
            }
    output = root / args.output
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

