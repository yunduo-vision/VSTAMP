from __future__ import annotations

from typing import Mapping

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .temporal_encoder import TemporalEncoder


class AlignedMultiscalePooling(nn.Module):
    def __init__(
        self,
        resolutions_ms: tuple[int, ...] = (100, 500, 2000),
        tokens: int = 30,
        dropout: float = 0.1,
        uniform_fusion: bool = False,
        epsilon: float = 1e-8,
    ) -> None:
        super().__init__()
        self.resolutions = tuple(int(value) for value in resolutions_ms)
        self.tokens = tokens
        self.uniform_fusion = uniform_fusion
        self.epsilon = epsilon
        self.encoders = nn.ModuleDict({str(r): TemporalEncoder(dropout=dropout) for r in self.resolutions})
        self.gates = nn.ModuleDict({str(r): nn.Linear(256, 1) for r in self.resolutions})
        self.projections = nn.ModuleDict({str(r): nn.Linear(256, 256) for r in self.resolutions})
        self.layer_norm = nn.LayerNorm(256)
        self.projection_head = nn.Sequential(
            nn.Linear(512, 512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, 512),
        )

    def forward(
        self,
        features: Mapping[int, Tensor],
        masks: Mapping[int, Tensor],
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Return ``z[B,512], U[B,M,256], r[B,M], alpha[B,M,D]`` (Eq. 4-7)."""
        encoded: list[Tensor] = []
        reliability: list[Tensor] = []
        logits: list[Tensor] = []
        projected: list[Tensor] = []
        for resolution in self.resolutions:
            tokens, valid = self.encoders[str(resolution)](features[resolution], masks[resolution], self.tokens)
            encoded.append(tokens)
            reliability.append(valid)
            logits.append(self.gates[str(resolution)](tokens).squeeze(-1))
            projected.append(self.projections[str(resolution)](tokens))
        reliability_tensor = torch.stack(reliability, dim=-1)
        if self.uniform_fusion:
            weights = reliability_tensor
        else:
            gate_tensor = torch.stack(logits, dim=-1)
            gate_tensor = gate_tensor - gate_tensor.max(dim=-1, keepdim=True).values
            weights = reliability_tensor * torch.exp(gate_tensor)
        alpha = weights / weights.sum(dim=-1, keepdim=True).clamp_min(self.epsilon)
        projected_tensor = torch.stack(projected, dim=-2)
        fused = (alpha.unsqueeze(-1) * projected_tensor).sum(dim=-2)
        fused = self.layer_norm(F.gelu(fused))
        validity = reliability_tensor.max(dim=-1).values
        weighted_mean = (validity.unsqueeze(-1) * fused).sum(dim=1) / validity.sum(dim=1, keepdim=True).clamp_min(self.epsilon)
        valid_tokens = validity > 0
        masked_for_max = fused.masked_fill(~valid_tokens.unsqueeze(-1), float("-inf"))
        maximum = masked_for_max.max(dim=1).values
        maximum = torch.where(valid_tokens.any(dim=1, keepdim=True), maximum, torch.zeros_like(maximum))
        global_input = torch.cat((weighted_mean, maximum), dim=-1)
        z = F.normalize(self.projection_head(global_input), dim=-1, eps=self.epsilon)
        return z, fused, validity, alpha

