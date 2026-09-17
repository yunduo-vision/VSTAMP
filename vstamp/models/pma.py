from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def _smax(candidates: list[Tensor], gamma: float) -> Tensor:
    return gamma * torch.logsumexp(torch.stack(candidates, dim=-1) / gamma, dim=-1)


def pma_forward(affinity: Tensor, valid: Tensor, gamma: float = 0.1, gap: float = 0.1) -> tuple[Tensor, list[list[Tensor]]]:
    """Compute Eq. (9-10) for ``affinity[B,M,N]`` without in-place autograd mutations."""
    if affinity.ndim != 3 or valid.shape != affinity.shape:
        raise ValueError("affinity and valid must have shape [B,M,N]")
    batch, rows, columns = affinity.shape
    zero = affinity.new_zeros(batch)
    dp: list[list[Tensor]] = [[zero for _ in range(columns + 1)] for _ in range(rows + 1)]
    masked = affinity.masked_fill(~valid, -1e4)
    for i in range(1, rows + 1):
        for j in range(1, columns + 1):
            dp[i][j] = _smax(
                [
                    zero,
                    dp[i - 1][j - 1] + masked[:, i - 1, j - 1],
                    dp[i - 1][j] - gap,
                    dp[i][j - 1] - gap,
                ],
                gamma,
            )
    cells = torch.stack([dp[i][j] for i in range(1, rows + 1) for j in range(1, columns + 1)], dim=-1)
    score = gamma * torch.logsumexp(cells / gamma, dim=-1)
    return score, dp


def pma_explicit(
    affinity: Tensor,
    valid: Tensor,
    gamma: float = 0.1,
    gap: float = 0.1,
) -> tuple[Tensor, Tensor]:
    """Compute PMA score and Eq. (11) via explicit reverse occupancy recursion."""
    score, dp = pma_forward(affinity, valid, gamma=gamma, gap=gap)
    batch, rows, columns = affinity.shape
    cells = torch.stack([dp[i][j] for i in range(1, rows + 1) for j in range(1, columns + 1)], dim=-1)
    end_probability = torch.softmax(cells / gamma, dim=-1).reshape(batch, rows, columns)
    adjoint: list[list[Tensor]] = [
        [affinity.new_zeros(batch) for _ in range(columns + 1)] for _ in range(rows + 1)
    ]
    mass_rows: list[list[Tensor | None]] = [[None for _ in range(columns)] for _ in range(rows)]
    masked = affinity.masked_fill(~valid, -1e4)
    zero = affinity.new_zeros(batch)
    for i in range(rows, 0, -1):
        for j in range(columns, 0, -1):
            occupancy = adjoint[i][j] + end_probability[:, i - 1, j - 1]
            candidates = torch.stack(
                [
                    zero,
                    dp[i - 1][j - 1] + masked[:, i - 1, j - 1],
                    dp[i - 1][j] - gap,
                    dp[i][j - 1] - gap,
                ],
                dim=-1,
            )
            transition = torch.softmax(candidates / gamma, dim=-1)
            diagonal = occupancy * transition[:, 1]
            mass_rows[i - 1][j - 1] = diagonal * valid[:, i - 1, j - 1].to(diagonal.dtype)
            if i > 1 and j > 1:
                adjoint[i - 1][j - 1] = adjoint[i - 1][j - 1] + diagonal
            if i > 1:
                adjoint[i - 1][j] = adjoint[i - 1][j] + occupancy * transition[:, 2]
            if j > 1:
                adjoint[i][j - 1] = adjoint[i][j - 1] + occupancy * transition[:, 3]
    mass = torch.stack([torch.stack([value for value in row if value is not None], dim=-1) for row in mass_rows], dim=1)
    return score, mass


def pma_autograd_reference(
    affinity: Tensor,
    valid: Tensor,
    gamma: float = 0.1,
    gap: float = 0.1,
    create_graph: bool = False,
) -> tuple[Tensor, Tensor]:
    with torch.enable_grad():
        reference = affinity if affinity.requires_grad else affinity.detach().requires_grad_(True)
        score, _ = pma_forward(reference, valid, gamma=gamma, gap=gap)
        mass = torch.autograd.grad(score.sum(), reference, create_graph=create_graph)[0]
    return score, mass


@dataclass
class PairMetrics:
    pair_score: Tensor
    global_similarity: Tensor
    local_similarity: Tensor
    overlap: Tensor
    pma_score: Tensor
    alignment_mass: Tensor
    affinity: Tensor
    local_cosine: Tensor


class PartialMonotoneAlignment(nn.Module):
    def __init__(
        self,
        input_dim: int = 256,
        alignment_dim: int = 128,
        match_offset: float = 0.2,
        gap: float = 0.1,
        gamma: float = 0.1,
        lambda_z: float = 0.5,
        epsilon: float = 1e-8,
    ) -> None:
        super().__init__()
        self.local_projection = nn.Linear(input_dim, alignment_dim)
        self.match_offset = match_offset
        self.gap = gap
        self.gamma = gamma
        self.lambda_z = lambda_z
        self.epsilon = epsilon

    def project(self, tokens: Tensor) -> Tensor:
        return F.normalize(self.local_projection(tokens), dim=-1, eps=self.epsilon)

    def forward(
        self,
        z_a: Tensor,
        tokens_a: Tensor,
        reliability_a: Tensor,
        z_b: Tensor,
        tokens_b: Tensor,
        reliability_b: Tensor,
    ) -> PairMetrics:
        """Compare paired batches and return Eq. (8-13,18) metrics."""
        local_a, local_b = self.project(tokens_a), self.project(tokens_b)
        cosine = torch.einsum("bmd,bnd->bmn", local_a, local_b)
        affinity = cosine - self.match_offset
        valid = (reliability_a > 0).unsqueeze(-1) & (reliability_b > 0).unsqueeze(-2)
        pma_score, mass = pma_explicit(affinity, valid, gamma=self.gamma, gap=self.gap)
        mass_total = mass.sum(dim=(-2, -1)).clamp_min(self.epsilon)
        local_similarity = (mass * cosine).sum(dim=(-2, -1)) / mass_total
        common_reliability = torch.minimum(reliability_a.unsqueeze(-1), reliability_b.unsqueeze(-2))
        smaller_mass = torch.minimum(reliability_a.sum(-1), reliability_b.sum(-1)).clamp_min(self.epsilon)
        overlap = ((mass * common_reliability).sum(dim=(-2, -1)) / smaller_mass).clamp(0.0, 1.0)
        global_similarity = (z_a * z_b).sum(dim=-1)
        pair_score = self.lambda_z * global_similarity + (1.0 - self.lambda_z) * local_similarity
        return PairMetrics(pair_score, global_similarity, local_similarity, overlap, pma_score, mass, affinity, cosine)

