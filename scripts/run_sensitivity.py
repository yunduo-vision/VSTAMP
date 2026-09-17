from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from vstamp.config import load_config
from vstamp.training import train


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run matched-seed VSTAMP sensitivity experiments")
    parser.add_argument("--config", required=True)
    parser.add_argument("--parameter", required=True, choices=("tokens", "gap_penalty", "lambda_z"))
    parser.add_argument("--values", nargs="+", type=float, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1])
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    base = load_config(args.config)
    for value in args.values:
        cfg = copy.deepcopy(base)
        if args.parameter == "tokens":
            if not float(value).is_integer():
                raise ValueError("Token counts must be integers")
            cfg["model"]["tokens"] = int(value)
        else:
            cfg["matching"][args.parameter] = float(value)
        value_label = str(value).replace(".", "p")
        cfg["output"]["experiment_name"] = f"sensitivity_{args.parameter}_{value_label}"
        for seed in args.seeds:
            print(train(cfg, seed, torch.device(args.device)))


if __name__ == "__main__":
    main()

