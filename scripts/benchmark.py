from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from vstamp.config import load_config
from vstamp.data.dataset import ProcessedDataset
from vstamp.evaluation.runtime import benchmark_model
from vstamp.models.vstamp import VSTAMP
from vstamp.utils.checkpoint import load_checkpoint


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark FP32 VSTAMP encoding and 5-way 2-shot matching")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--output", help="Optional JSON output path")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    cfg = load_config(args.config)
    device = torch.device(args.device)
    model = VSTAMP(cfg).to(device)
    model.load_state_dict(load_checkpoint(args.checkpoint, device)["model"])
    dataset = ProcessedDataset(cfg["data"]["processed_root"], split="test")
    session_id = dataset.records[0].session_id
    features, masks = dataset.tensors([session_id], device)
    metrics = benchmark_model(model, features, masks, args.warmup, args.iterations, device)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()

