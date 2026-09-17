from __future__ import annotations

import torch
from torch import Tensor
from torch.nn import functional as F


def supervised_contrastive_loss(features: Tensor, labels: Tensor, temperature: float = 0.07) -> Tensor:
    """Standard supervised contrastive loss with self-pairs excluded."""
    normalized = F.normalize(features, dim=-1)
    logits = normalized @ normalized.transpose(0, 1) / temperature
    logits = logits - logits.max(dim=1, keepdim=True).values.detach()
    eye = torch.eye(len(features), dtype=torch.bool, device=features.device)
    positives = labels.view(-1, 1).eq(labels.view(1, -1)) & ~eye
    valid_anchor = positives.any(dim=1)
    if not valid_anchor.any():
        return features.sum() * 0.0
    exp_logits = torch.exp(logits) * (~eye).to(logits.dtype)
    log_probability = logits - torch.log(exp_logits.sum(dim=1, keepdim=True).clamp_min(1e-8))
    positive_mean = (log_probability * positives.to(logits.dtype)).sum(dim=1) / positives.sum(dim=1).clamp_min(1)
    return -positive_mean[valid_anchor].mean()


def alignment_loss(mass: Tensor, target: Tensor, valid_columns: Tensor, epsilon: float = 1e-8) -> Tensor:
    """Column-normalized correspondence cross-entropy from Eq. (17)."""
    probability = mass / mass.sum(dim=-2, keepdim=True).clamp_min(epsilon)
    per_column = -(target * torch.log(probability.clamp_min(epsilon))).sum(dim=-2)
    valid_count = valid_columns.sum().clamp_min(1)
    return (per_column * valid_columns.to(per_column.dtype)).sum() / valid_count


def episodic_loss(candidate_scores: Tensor, targets: Tensor, temperature: float = 0.1) -> Tensor:
    return F.cross_entropy(candidate_scores / temperature, targets)


def assert_finite(name: str, value: Tensor, context: str = "") -> None:
    if not torch.isfinite(value).all():
        suffix = f" ({context})" if context else ""
        raise RuntimeError(f"Non-finite {name}{suffix}")

