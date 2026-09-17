from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from vstamp.config import load_config
from vstamp.data.dataset import ProcessedDataset
from vstamp.evaluation.correspondence import evaluate_correspondence
from vstamp.evaluation.open_set import evaluate_open_set
from vstamp.evaluation.protocols import fixed_episodes
from vstamp.evaluation.real_overlap import evaluate_real_overlap
from vstamp.evaluation.recognition import evaluate_episodes
from vstamp.evaluation.truncation import evaluate_query_truncation
from vstamp.models.vstamp import VSTAMP
from vstamp.utils.checkpoint import load_checkpoint
from vstamp.utils.metrics import mean_sample_std


PROTOCOLS = (
    "cross_mode",
    "same_mode",
    "balanced_3shot",
    "random_5shot",
    "cross_bandwidth",
    "bandwidth_pair",
    "query_truncation",
    "correspondence",
    "unseen_bandwidth",
    "open_set",
    "real_overlap",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a VSTAMP checkpoint on fixed few-shot protocols")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--protocol", required=True, choices=PROTOCOLS)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--episodes", type=int, help="Override the paper protocol episode count")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-dir", help="Defaults to a results directory beside the checkpoint")
    parser.add_argument("--support-bandwidth", choices=("bw1", "bw2", "bw4", "bw8"))
    parser.add_argument("--query-bandwidth", choices=("bw1", "bw2", "bw4", "bw8"))
    return parser


def _save_predictions(output: Path, rows: list[dict[str, object]]) -> None:
    frame = pd.DataFrame(rows)
    frame.to_csv(output / "episode_results.csv", index=False)
    if not frame.empty:
        np.savez_compressed(
            output / "test_predictions.npz",
            session_id=frame["session_id"].astype(str).to_numpy(),
            target=frame["target"].astype(str).to_numpy(),
            prediction=frame["prediction"].astype(str).to_numpy(),
            score=frame["max_score"].to_numpy(dtype=np.float32),
        )


def main() -> None:
    args = build_parser().parse_args()
    cfg = load_config(args.config)
    device = torch.device(args.device)
    checkpoint = load_checkpoint(args.checkpoint, device)
    model = VSTAMP(cfg).to(device)
    model.load_state_dict(checkpoint["model"])
    dataset = ProcessedDataset(cfg["data"]["processed_root"], split="test")
    output = (
        Path(args.output_dir).resolve()
        if args.output_dir
        else Path(args.checkpoint).resolve().parent / "evaluation" / args.protocol
    )
    output.mkdir(parents=True, exist_ok=True)
    protocol = args.protocol
    metrics: dict[str, object]
    rows: list[dict[str, object]] = []
    if protocol in {
        "cross_mode",
        "same_mode",
        "balanced_3shot",
        "random_5shot",
        "cross_bandwidth",
        "bandwidth_pair",
    }:
        episode_protocol = protocol
        if protocol == "bandwidth_pair":
            if not args.support_bandwidth or not args.query_bandwidth:
                raise ValueError(
                    "bandwidth_pair requires --support-bandwidth and --query-bandwidth"
                )
            if args.support_bandwidth == args.query_bandwidth:
                raise ValueError("The bandwidth-pair diagnostic requires distinct conditions")
            episode_protocol = f"bandwidth_{args.support_bandwidth}_{args.query_bandwidth}"
        default_count = 500 if cfg["dataset"]["name"] == "ydms" and protocol == "cross_mode" else 300
        count = args.episodes or default_count
        episodes = fixed_episodes(
            dataset.records,
            episode_protocol,
            count,
            args.seed,
            str(cfg["dataset"]["name"]),
            Path(cfg["data"]["processed_root"]) / "episodes",
        )
        metrics, rows = evaluate_episodes(model, dataset, episodes, device)
        _save_predictions(output, rows)
    elif protocol == "query_truncation":
        count = args.episodes or 300
        episodes = fixed_episodes(
            dataset.records,
            "cross_mode",
            count,
            args.seed,
            str(cfg["dataset"]["name"]),
            Path(cfg["data"]["processed_root"]) / "episodes",
        )
        curve = evaluate_query_truncation(
            model,
            dataset,
            episodes,
            [float(value) for value in cfg["evaluation"]["truncation_fractions"]],
            device,
        )
        pd.DataFrame(curve).to_csv(output / "query_truncation.csv", index=False)
        metrics = {"curve": curve}
    elif protocol == "correspondence":
        metrics = evaluate_correspondence(
            model,
            dataset,
            args.episodes or int(cfg["evaluation"]["correspondence_pairs"]),
            args.seed,
            device,
        )
    elif protocol == "unseen_bandwidth":
        bandwidth = str(cfg["dataset"].get("held_out_bandwidth", ""))
        if not bandwidth:
            raise ValueError("unseen_bandwidth requires dataset.held_out_bandwidth in the config")
        episodes = fixed_episodes(
            dataset.records,
            f"unseen_{bandwidth}",
            args.episodes or 300,
            args.seed,
            str(cfg["dataset"]["name"]),
            Path(cfg["data"]["processed_root"]) / "episodes",
        )
        metrics, rows = evaluate_episodes(model, dataset, episodes, device)
        _save_predictions(output, rows)
    elif protocol == "open_set":
        partition_metrics = evaluate_open_set(
            model,
            dataset,
            int(cfg["evaluation"]["open_set_partitions"]),
            args.seed,
            device,
        )
        auroc = mean_sample_std(row["auroc"] for row in partition_metrics)
        fpr = mean_sample_std(row["fpr_at_95_tpr"] for row in partition_metrics)
        metrics = {
            "partitions": partition_metrics,
            "auroc_mean": auroc[0],
            "auroc_sample_std": auroc[1],
            "fpr_at_95_tpr_mean": fpr[0],
            "fpr_at_95_tpr_sample_std": fpr[1],
        }
    elif protocol == "real_overlap":
        metrics = evaluate_real_overlap(
            model,
            dataset,
            args.seed,
            int(cfg["evaluation"]["bootstrap_samples"]),
            int(cfg["evaluation"]["real_overlap_windows"]),
            int(cfg["evaluation"]["real_overlap_pairs"]),
            device,
        )
    else:
        raise AssertionError(protocol)
    with (output / "metrics.json").open("w", encoding="utf-8") as stream:
        json.dump(metrics, stream, indent=2)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
