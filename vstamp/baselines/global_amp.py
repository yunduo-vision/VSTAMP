from __future__ import annotations

from typing import Mapping

from torch import Tensor, nn

from ..models.amp import AlignedMultiscalePooling


class GlobalAMP(nn.Module):
    """Global-only AMP encoder used by the SupportMax control."""

    def __init__(self, cfg: Mapping[str, object]) -> None:
        super().__init__()
        self.encoder = AlignedMultiscalePooling(
            resolutions_ms=tuple(cfg["data"]["resolutions_ms"]),
            tokens=int(cfg["model"]["tokens"]),
            dropout=float(cfg["model"]["dropout"]),
            epsilon=float(cfg["numerics"]["epsilon"]),
        )

    def forward(self, features: Mapping[int, Tensor], masks: Mapping[int, Tensor]) -> Tensor:
        z, _, _, _ = self.encoder(features, masks)
        return z

