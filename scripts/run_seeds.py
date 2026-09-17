from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from vstamp.config import load_config
from vstamp.training import train


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run matched VSTAMP training seeds")
    parser.add_argument("--config", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    cfg = load_config(args.config)
    for seed in args.seeds:
        print(train(cfg, seed, torch.device(args.device)))


if __name__ == "__main__":
    main()

