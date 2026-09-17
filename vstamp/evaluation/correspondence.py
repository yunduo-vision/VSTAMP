from __future__ import annotations

import numpy as np
import torch
from torch import Tensor

from ..data.dataset import ProcessedDataset
from ..data.partial_views import linear_interpolate_tokens, piecewise_warp, token_centers
from ..models.vstamp import Fingerprint, VSTAMP


def _interval_mask(
    features: dict[int, Tensor], masks: dict[int, Tensor], start: Tensor, end: Tensor
) -> tuple[dict[int, Tensor], dict[int, Tensor]]:
    output_x: dict[int, Tensor] = {}
    output_m: dict[int, Tensor] = {}
    for resolution, values in features.items():
        length = values.shape[1]
        centers = (torch.arange(length, device=values.device, dtype=values.dtype) + 0.5) * 60.0 / length
        visible = (centers.unsqueeze(0) >= start.unsqueeze(1)) & (centers.unsqueeze(0) <= end.unsqueeze(1))
        output_x[resolution] = values * visible.unsqueeze(-1)
        output_m[resolution] = masks[resolution] * visible
    return output_x, output_m


def evaluate_correspondence(
    model: VSTAMP,
    dataset: ProcessedDataset,
    pairs: int,
    seed: int,
    device: torch.device,
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    session_ids = list(dataset.by_id)
    if not session_ids:
        raise ValueError("No test sessions for correspondence evaluation")
    errors: list[float] = []
    within: list[float] = []
    overlap_errors: list[float] = []
    model.eval()
    for _ in range(pairs):
        session_id = session_ids[int(rng.integers(len(session_ids)))]
        features, masks = dataset.tensors([session_id], device)
        starts = torch.tensor(rng.uniform(0.0, 18.0, size=2), device=device, dtype=torch.float32)
        lengths = torch.tensor(rng.uniform(24.0, 60.0, size=2), device=device, dtype=torch.float32)
        ends = torch.minimum(starts + lengths, torch.tensor(60.0, device=device))
        etas = torch.tensor(rng.uniform(-0.3, 0.3, size=2), device=device, dtype=torch.float32)
        fingerprints: list[Fingerprint] = []
        coordinates: list[Tensor] = []
        for view in range(2):
            x, m = _interval_mask(features, masks, starts[view : view + 1], ends[view : view + 1])
            base = model.encode(x, m)
            centers = token_centers(base.tokens.shape[1], 60.0, device, base.tokens.dtype)
            coordinate = piecewise_warp(centers, etas[view], 60.0)
            warped_tokens = linear_interpolate_tokens(base.tokens, coordinate, 60.0)
            warped_reliability = linear_interpolate_tokens(base.reliability, coordinate, 60.0)
            fingerprints.append(Fingerprint(base.z, warped_tokens, warped_reliability))
            coordinates.append(coordinate.squeeze(0) if coordinate.ndim == 2 else coordinate)
        with torch.no_grad():
            metrics = model.compare(fingerprints[0], fingerprints[1])
        mass = metrics.alignment_mass[0]
        probability = mass / mass.sum(dim=0, keepdim=True).clamp_min(1e-8)
        predicted = (probability * coordinates[0].unsqueeze(1)).sum(dim=0)
        common_start = torch.maximum(starts[0], starts[1])
        common_end = torch.minimum(ends[0], ends[1])
        valid = (
            (coordinates[1] >= common_start)
            & (coordinates[1] <= common_end)
            & (fingerprints[1].reliability[0] > 0)
            & (mass.sum(dim=0) > 1e-8)
        )
        if valid.any():
            absolute = (predicted[valid] - coordinates[1][valid]).abs()
            errors.extend(absolute.cpu().tolist())
            within.extend((absolute <= 2.0).float().cpu().tolist())
        intersection = (common_end - common_start).clamp_min(0.0)
        reference_overlap = intersection / torch.minimum(ends[0] - starts[0], ends[1] - starts[1]).clamp_min(1e-8)
        overlap_errors.append(float((metrics.overlap[0] - reference_overlap).abs().cpu()))
    if not errors:
        raise RuntimeError("No valid common-support correspondence tokens were produced")
    return {
        "align_mae_seconds": float(np.mean(errors)),
        "align_at_2s": float(np.mean(within)),
        "overlap_mae": float(np.mean(overlap_errors)),
        "evaluated_tokens": len(errors),
    }

