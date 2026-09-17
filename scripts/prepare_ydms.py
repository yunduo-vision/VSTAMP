from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from vstamp.data.preprocess import prepare_dataset


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare the YDMS mobile YouTube dataset for VSTAMP")
    parser.add_argument("--data-root", required=True, help="YDMS root containing measurement directories")
    parser.add_argument("--output-root", required=True, help="Directory for processed features and split metadata")
    parser.add_argument("--seed", type=int, default=0, help="Identity-split seed")
    parser.add_argument(
        "--allow-nonstandard-identity-count",
        action="store_true",
        help="Allow development subsets; formal experiments require 192 usable identities",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary = prepare_dataset(
        "ydms",
        args.data_root,
        args.output_root,
        seed=args.seed,
        strict_counts=not args.allow_nonstandard_identity_count,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

