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
    parser = argparse.ArgumentParser(description="Train VSTAMP with episode-based optimization")
    parser.add_argument("--config", required=True, help="YAML experiment configuration")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--episodes", type=int, help="Override episode count for controlled development checks")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    cfg = load_config(args.config)
    if args.episodes is not None:
        if args.episodes <= 0:
            raise ValueError("--episodes must be positive")
        cfg["training"]["episodes"] = args.episodes
    output = train(cfg, args.seed, torch.device(args.device))
    print(output)


if __name__ == "__main__":
    main()

