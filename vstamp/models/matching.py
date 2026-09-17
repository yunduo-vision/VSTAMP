from __future__ import annotations

import torch
from torch import Tensor


def compatibility_responsibilities(
    overlap: Tensor,
    local_similarity: Tensor,
    beta_overlap: float = 1.0,
    beta_similarity: float = 1.0,
    temperature: float = 0.2,
) -> Tensor:
    compatibility = beta_overlap * overlap + beta_similarity * local_similarity
    return torch.softmax(compatibility / temperature, dim=-1)


def aggregate_candidates(
    pair_scores: Tensor,
    overlap: Tensor,
    local_similarity: Tensor,
    method: str = "compatibility",
    beta_overlap: float = 1.0,
    beta_similarity: float = 1.0,
    compatibility_temperature: float = 0.2,
    score_temperature: float = 0.1,
    epsilon: float = 1e-8,
) -> tuple[Tensor, Tensor]:
    """Aggregate ``[...,K]`` support evidence into candidate scores (Eq. 19-20)."""
    if method == "mean":
        responsibilities = torch.full_like(pair_scores, 1.0 / pair_scores.shape[-1])
        return pair_scores.mean(dim=-1), responsibilities
    if method == "support_max":
        index = pair_scores.argmax(dim=-1, keepdim=True)
        responsibilities = torch.zeros_like(pair_scores).scatter_(-1, index, 1.0)
        return pair_scores.max(dim=-1).values, responsibilities
    if method == "uniform_lse":
        responsibilities = torch.full_like(pair_scores, 1.0 / pair_scores.shape[-1])
    elif method == "similarity_only":
        responsibilities = compatibility_responsibilities(
            overlap, local_similarity, 0.0, beta_similarity, compatibility_temperature
        )
    elif method == "overlap_only":
        responsibilities = compatibility_responsibilities(
            overlap, local_similarity, beta_overlap, 0.0, compatibility_temperature
        )
    elif method == "compatibility":
        responsibilities = compatibility_responsibilities(
            overlap,
            local_similarity,
            beta_overlap,
            beta_similarity,
            compatibility_temperature,
        )
    else:
        raise ValueError(f"Unknown aggregation method: {method}")
    log_terms = torch.log(responsibilities.clamp_min(epsilon)) + pair_scores / score_temperature
    candidate = score_temperature * torch.logsumexp(log_terms, dim=-1)
    return candidate, responsibilities

