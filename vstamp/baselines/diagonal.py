from __future__ import annotations

import torch
from torch import Tensor

from ..models.pma import PairMetrics, PartialMonotoneAlignment


class DiagonalAlignment(PartialMonotoneAlignment):
    def forward(
        self,
        z_a: Tensor,
        tokens_a: Tensor,
        reliability_a: Tensor,
        z_b: Tensor,
        tokens_b: Tensor,
        reliability_b: Tensor,
    ) -> PairMetrics:
        local_a, local_b = self.project(tokens_a), self.project(tokens_b)
        cosine = torch.einsum("bmd,bnd->bmn", local_a, local_b)
        size = min(tokens_a.shape[1], tokens_b.shape[1])
        diagonal = torch.eye(tokens_a.shape[1], tokens_b.shape[1], device=tokens_a.device, dtype=tokens_a.dtype)
        valid = (reliability_a > 0).unsqueeze(-1) & (reliability_b > 0).unsqueeze(-2)
        mass = diagonal.unsqueeze(0) * valid.to(tokens_a.dtype)
        denominator = mass.sum(dim=(-2, -1)).clamp_min(self.epsilon)
        local = (mass * cosine).sum(dim=(-2, -1)) / denominator
        global_similarity = (z_a * z_b).sum(dim=-1)
        pair = self.lambda_z * global_similarity + (1.0 - self.lambda_z) * local
        overlap = torch.zeros_like(local)
        return PairMetrics(pair, global_similarity, local, overlap, local, mass, cosine - self.match_offset, cosine)

