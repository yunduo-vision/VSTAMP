from __future__ import annotations

import time

import torch

from ..models.vstamp import Fingerprint, VSTAMP


def _measure(operation, warmup: int, iterations: int, device: torch.device) -> float:
    for _ in range(warmup):
        operation()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    start = time.perf_counter()
    for _ in range(iterations):
        operation()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return (time.perf_counter() - start) * 1000.0 / iterations


def benchmark_model(
    model: VSTAMP,
    features: dict[int, torch.Tensor],
    masks: dict[int, torch.Tensor],
    warmup: int,
    iterations: int,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    with torch.no_grad():
        support_single = model.encode(features, masks)
        supports = Fingerprint(
            support_single.z.expand(10, -1).reshape(5, 2, -1).contiguous(),
            support_single.tokens.expand(10, -1, -1).reshape(5, 2, *support_single.tokens.shape[1:]).contiguous(),
            support_single.reliability.expand(10, -1).reshape(5, 2, -1).contiguous(),
        )

        def encode() -> None:
            model.encode(features, masks)

        query = model.encode(features, masks)

        def match() -> None:
            model.score_candidates(query, supports)

        encoding_ms = _measure(encode, warmup, iterations, device)
        matching_ms = _measure(match, warmup, iterations, device)
    parameters = sum(parameter.numel() for parameter in model.parameters())
    storage = (query.z.numel() + query.tokens.numel() + query.reliability.numel()) * 4
    return {
        "parameters": float(parameters),
        "support_fingerprint_kib": storage / 1024.0,
        "encoding_ms": encoding_ms,
        "matching_5way_2shot_ms": matching_ms,
    }

