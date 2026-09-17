from __future__ import annotations

import torch

from ..data.dataset import ProcessedDataset
from ..data.episode_sampler import Episode
from ..models.vstamp import VSTAMP
from .recognition import evaluate_episodes


def evaluate_query_truncation(
    model: VSTAMP,
    dataset: ProcessedDataset,
    episodes: list[Episode],
    fractions: list[float],
    device: torch.device,
) -> list[dict[str, float]]:
    results = []
    for fraction in fractions:
        metrics, _ = evaluate_episodes(model, dataset, episodes, device, retained_fraction=fraction)
        results.append(
            {
                "retained_fraction": float(fraction),
                "seconds": float(fraction * 60.0),
                "accuracy": float(metrics["accuracy"]),
            }
        )
    return results

