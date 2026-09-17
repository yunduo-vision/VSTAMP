from __future__ import annotations

from collections import defaultdict

import numpy as np
import torch
from scipy.stats import spearmanr

from ..data.dataset import ProcessedDataset
from ..models.vstamp import VSTAMP


def evaluate_real_overlap(
    model: VSTAMP,
    dataset: ProcessedDataset,
    seed: int,
    bootstrap_samples: int,
    window_limit: int,
    pair_limit: int,
    device: torch.device,
) -> dict[str, object]:
    missing = [
        record.session_id
        for record in dataset.records
        if "media_start" not in record.extra or "media_end" not in record.extra
    ]
    if missing:
        raise ValueError(
            "Real-session overlap requires evaluation-only media_start and media_end fields derived from "
            "player playback-progress/buffer logs; missing for sessions including " + ", ".join(missing[:5])
        )
    rng = np.random.default_rng(seed)
    selected_records = list(dataset.records)
    if len(selected_records) > window_limit:
        selected = rng.choice(len(selected_records), size=window_limit, replace=False)
        selected_records = [selected_records[int(index)] for index in selected]
    grouped: dict[str, list] = defaultdict(list)
    for record in selected_records:
        grouped[record.video_id].append(record)
    candidate_pairs = []
    for rows in grouped.values():
        for left_index, left in enumerate(rows):
            for right in rows[left_index + 1 :]:
                if left.bandwidth != right.bandwidth:
                    candidate_pairs.append((left, right))
    if len(candidate_pairs) > pair_limit:
        selected = rng.choice(len(candidate_pairs), size=pair_limit, replace=False)
        candidate_pairs = [candidate_pairs[int(index)] for index in selected]
    estimated: list[float] = []
    reference: list[float] = []
    model.eval()
    for left, right in candidate_pairs:
        features, masks = dataset.tensors([left.session_id, right.session_id], device)
        with torch.no_grad():
            fingerprints = model.encode(features, masks)
            metrics = model.compare(
                type(fingerprints)(
                    fingerprints.z[:1], fingerprints.tokens[:1], fingerprints.reliability[:1]
                ),
                type(fingerprints)(
                    fingerprints.z[1:], fingerprints.tokens[1:], fingerprints.reliability[1:]
                ),
            )
        left_start, left_end = float(left.extra["media_start"]), float(left.extra["media_end"])
        right_start, right_end = float(right.extra["media_start"]), float(right.extra["media_end"])
        common = max(0.0, min(left_end, right_end) - max(left_start, right_start))
        denominator = max(min(left_end - left_start, right_end - right_start), 1e-8)
        estimated.append(float(metrics.overlap[0].cpu()))
        reference.append(common / denominator)
    if not estimated:
        raise ValueError("No same-video cross-bandwidth pairs with media intervals were found")
    estimated_array = np.asarray(estimated)
    reference_array = np.asarray(reference)
    rho = float(spearmanr(estimated_array, reference_array).statistic)
    mae = float(np.mean(np.abs(estimated_array - reference_array)))
    rho_boot, mae_boot = [], []
    for _ in range(bootstrap_samples):
        indices = rng.integers(0, len(estimated_array), len(estimated_array))
        rho_boot.append(float(spearmanr(estimated_array[indices], reference_array[indices]).statistic))
        mae_boot.append(float(np.mean(np.abs(estimated_array[indices] - reference_array[indices]))))
    return {
        "pairs": len(estimated),
        "spearman_rho": rho,
        "spearman_95_ci": np.nanpercentile(rho_boot, [2.5, 97.5]).tolist(),
        "mae": mae,
        "mae_95_ci": np.percentile(mae_boot, [2.5, 97.5]).tolist(),
    }
