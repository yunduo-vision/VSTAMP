from __future__ import annotations

import torch
from torch import Tensor, nn


class TemporalEncoder(nn.Module):
    def __init__(self, input_channels: int = 6, dropout: float = 0.1) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv1d(input_channels, 128, kernel_size=5, stride=1, padding=2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(128, 256, kernel_size=5, stride=1, padding=2),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, features: Tensor, mask: Tensor, tokens: int) -> tuple[Tensor, Tensor]:
        """Encode ``features[B,L,5]`` and return tokens ``[B,M,256]`` and reliability ``[B,M]``."""
        if features.ndim != 3 or features.shape[-1] != 5:
            raise ValueError("features must have shape [B,L,5]")
        if mask.shape != features.shape[:2]:
            raise ValueError("mask must have shape [B,L]")
        masked = features * mask.unsqueeze(-1)
        inputs = torch.cat((masked, mask.unsqueeze(-1)), dim=-1).transpose(1, 2)
        encoded = self.network(inputs)
        pooled = torch.nn.functional.adaptive_avg_pool1d(encoded, tokens).transpose(1, 2)
        reliability = torch.nn.functional.adaptive_avg_pool1d(mask.unsqueeze(1), tokens).squeeze(1)
        return pooled, reliability.clamp(0.0, 1.0)

