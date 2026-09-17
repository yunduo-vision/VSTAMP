from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from vstamp.data.preprocess import prepare_dataset


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare the LongEnough variable-bandwidth subset for VSTAMP")
    parser.add_argument("--data-root", required=True, help="Extracted LongEnough-variable dataset root")
    parser.add_argument("--output-root", required=True, help="Directory for processed features and split metadata")
    parser.add_argument("--seed", type=int, default=0, help="Identity-split and clustering seed")
    parser.add_argument("--held-out-bandwidth", choices=("", "bw1", "bw2", "bw4", "bw8"), default="")
    parser.add_argument(
        "--allow-nonstandard-identity-count",
        action="store_true",
        help="Allow development subsets; formal experiments require all 100 identities",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary = prepare_dataset(
        "longenough",
        args.data_root,
        args.output_root,
        seed=args.seed,
        strict_counts=not args.allow_nonstandard_identity_count,
        held_out_bandwidth=args.held_out_bandwidth,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

