from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch
from torch import Tensor, nn

from ..baselines.diagonal import DiagonalAlignment
from ..baselines.soft_dtw import SoftDTWAlignment
from .amp import AlignedMultiscalePooling
from .matching import aggregate_candidates
from .pma import PairMetrics, PartialMonotoneAlignment


@dataclass
class Fingerprint:
    z: Tensor
    tokens: Tensor
    reliability: Tensor
    scale_weights: Tensor | None = None


class VSTAMP(nn.Module):
    def __init__(self, cfg: Mapping[str, object]) -> None:
        super().__init__()
        model_cfg = cfg["model"]
        matching_cfg = cfg["matching"]
        ablation_cfg = cfg.get("ablation", {})
        self.amp = AlignedMultiscalePooling(
            resolutions_ms=tuple(cfg["data"]["resolutions_ms"]),
            tokens=int(model_cfg["tokens"]),
            dropout=float(model_cfg["dropout"]),
            uniform_fusion=bool(ablation_cfg.get("uniform_scale_fusion", False)),
            epsilon=float(cfg["numerics"]["epsilon"]),
        )
        alignment_kwargs = dict(
            alignment_dim=int(model_cfg["alignment_dim"]),
            match_offset=float(matching_cfg["match_offset"]),
            gap=float(matching_cfg["gap_penalty"]),
            gamma=float(matching_cfg["pma_gamma"]),
            lambda_z=float(matching_cfg["lambda_z"]),
            epsilon=float(cfg["numerics"]["epsilon"]),
        )
        operator = str(matching_cfg.get("operator", "pma"))
        if operator == "pma":
            self.alignment = PartialMonotoneAlignment(**alignment_kwargs)
        elif operator == "diagonal":
            self.alignment = DiagonalAlignment(**alignment_kwargs)
        elif operator == "soft_dtw":
            self.alignment = SoftDTWAlignment(**alignment_kwargs)
        else:
            raise ValueError(f"Unknown alignment operator: {operator}")
        self.operator = operator
        self.global_only = bool(ablation_cfg.get("global_only", False))
        self.aggregation = str(matching_cfg["aggregation"])
        self.beta_overlap = float(matching_cfg["beta_overlap"])
        self.beta_similarity = float(matching_cfg["beta_similarity"])
        self.compatibility_temperature = float(matching_cfg["compatibility_temperature"])
        self.score_temperature = float(matching_cfg["score_temperature"])
        self.epsilon = float(cfg["numerics"]["epsilon"])

    def encode(self, features: Mapping[int, Tensor], masks: Mapping[int, Tensor]) -> Fingerprint:
        z, tokens, reliability, weights = self.amp(features, masks)
        return Fingerprint(z, tokens, reliability, weights)

    def compare(self, left: Fingerprint, right: Fingerprint) -> PairMetrics:
        return self.alignment(
            left.z,
            left.tokens,
            left.reliability,
            right.z,
            right.tokens,
            right.reliability,
        )

    def score_candidates(
        self,
        query: Fingerprint,
        supports: Fingerprint,
    ) -> tuple[Tensor, Tensor, PairMetrics]:
        """Score query ``[Q]`` against supports ``[W,K]`` and return ``[Q,W]``."""
        ways, shots = supports.z.shape[:2]
        queries = query.z.shape[0]
        q_z = query.z[:, None, None, :].expand(-1, ways, shots, -1).reshape(-1, query.z.shape[-1])
        s_z = supports.z[None, :, :, :].expand(queries, -1, -1, -1).reshape(-1, supports.z.shape[-1])
        if self.global_only:
            global_flat = (q_z * s_z).sum(dim=-1)
            pair = global_flat.reshape(queries, ways, shots)
            zeros = torch.zeros_like(pair)
            candidate, responsibilities = aggregate_candidates(
                pair,
                zeros,
                zeros,
                method=self.aggregation,
                beta_overlap=0.0,
                beta_similarity=0.0,
                compatibility_temperature=self.compatibility_temperature,
                score_temperature=self.score_temperature,
                epsilon=self.epsilon,
            )
            dummy_matrix = global_flat.new_zeros(global_flat.shape[0], 1, 1)
            metrics = PairMetrics(
                global_flat,
                global_flat,
                torch.zeros_like(global_flat),
                torch.zeros_like(global_flat),
                torch.zeros_like(global_flat),
                dummy_matrix,
                dummy_matrix,
                dummy_matrix,
            )
            return candidate, responsibilities, metrics
        q_u = query.tokens[:, None, None, :, :].expand(-1, ways, shots, -1, -1).reshape(-1, *query.tokens.shape[1:])
        q_r = query.reliability[:, None, None, :].expand(-1, ways, shots, -1).reshape(-1, query.reliability.shape[-1])
        s_u = supports.tokens[None, :, :, :, :].expand(queries, -1, -1, -1, -1).reshape(-1, *supports.tokens.shape[2:])
        s_r = supports.reliability[None, :, :, :].expand(queries, -1, -1, -1).reshape(-1, supports.reliability.shape[-1])
        metrics = self.alignment(q_z, q_u, q_r, s_z, s_u, s_r)
        pair = metrics.pair_score.reshape(queries, ways, shots)
        overlap = metrics.overlap.reshape(queries, ways, shots)
        local = metrics.local_similarity.reshape(queries, ways, shots)
        candidate, responsibilities = aggregate_candidates(
            pair,
            overlap,
            local,
            method=self.aggregation,
            beta_overlap=self.beta_overlap,
            beta_similarity=self.beta_similarity,
            compatibility_temperature=self.compatibility_temperature,
            score_temperature=self.score_temperature,
            epsilon=self.epsilon,
        )
        return candidate, responsibilities, metrics
