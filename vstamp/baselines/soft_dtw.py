from __future__ import annotations

import torch
from torch import Tensor

from ..models.pma import PairMetrics, PartialMonotoneAlignment


def soft_dtw_score(cosine: Tensor, valid: Tensor, gamma: float) -> Tensor:
    batch, rows, columns = cosine.shape
    inf = cosine.new_full((batch,), 1e4)
    zero = cosine.new_zeros(batch)
    dp: list[list[Tensor]] = [[inf for _ in range(columns + 1)] for _ in range(rows + 1)]
    dp[0][0] = zero
    cost = (1.0 - cosine).masked_fill(~valid, 1e4)
    for i in range(1, rows + 1):
        for j in range(1, columns + 1):
            previous = torch.stack((dp[i - 1][j - 1], dp[i - 1][j], dp[i][j - 1]), dim=-1)
            soft_min = -gamma * torch.logsumexp(-previous / gamma, dim=-1)
            dp[i][j] = cost[:, i - 1, j - 1] + soft_min
    return -dp[rows][columns]


class SoftDTWAlignment(PartialMonotoneAlignment):
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
        valid = (reliability_a > 0).unsqueeze(-1) & (reliability_b > 0).unsqueeze(-2)
        with torch.enable_grad():
            reference = cosine if cosine.requires_grad else cosine.detach().requires_grad_(True)
            score = soft_dtw_score(reference, valid, self.gamma)
            mass = torch.autograd.grad(score.sum(), reference, create_graph=self.training)[0]
        denominator = mass.sum(dim=(-2, -1)).clamp_min(self.epsilon)
        local = (mass * cosine).sum(dim=(-2, -1)) / denominator
        global_similarity = (z_a * z_b).sum(dim=-1)
        pair = self.lambda_z * global_similarity + (1.0 - self.lambda_z) * local
        overlap = torch.zeros_like(local)
        return PairMetrics(pair, global_similarity, local, overlap, score, mass, cosine - self.match_offset, cosine)

