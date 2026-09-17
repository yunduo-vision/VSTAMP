from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from vstamp.config import load_config
from vstamp.training import train


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Select Soft-DTW smoothing on validation identities")
    parser.add_argument("--config", default="configs/ablations/soft_dtw.yaml")
    parser.add_argument("--gammas", nargs="+", type=float, default=[0.05, 0.1, 0.2, 0.5, 1.0])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", default="outputs/soft_dtw_selection.json")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    base = load_config(args.config)
    validation: dict[str, float] = {}
    for gamma in args.gammas:
        cfg = copy.deepcopy(base)
        cfg["matching"]["pma_gamma"] = gamma
        cfg["output"]["experiment_name"] = f"soft_dtw_gamma_{str(gamma).replace('.', 'p')}"
        run = train(cfg, args.seed, torch.device(args.device))
        metrics = json.loads((run / "metrics.json").read_text(encoding="utf-8"))
        validation[str(gamma)] = float(metrics["best_validation_accuracy"])
    selected = max(validation, key=validation.get)
    result = {"selected_gamma": float(selected), "validation_accuracy": validation}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

