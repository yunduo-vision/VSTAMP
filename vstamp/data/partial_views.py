from __future__ import annotations

from typing import Mapping

import torch
from torch import Tensor


def prefix_retention(
    features: Mapping[int, Tensor],
    masks: Mapping[int, Tensor],
    rho: Tensor | float,
    observation_seconds: float = 60.0,
) -> tuple[dict[int, Tensor], dict[int, Tensor]]:
    """Apply Eq. (14) with one retention fraction shared across scales."""
    output_x: dict[int, Tensor] = {}
    output_m: dict[int, Tensor] = {}
    for resolution, values in features.items():
        mask = masks[resolution]
        length = values.shape[-2]
        centers = (torch.arange(length, device=values.device, dtype=values.dtype) + 0.5) * observation_seconds / length
        threshold = torch.as_tensor(rho, device=values.device, dtype=values.dtype) * observation_seconds
        while threshold.ndim < values.ndim - 1:
            threshold = threshold.unsqueeze(-1)
        visible = centers <= threshold
        while visible.ndim < values.ndim:
            visible = visible.unsqueeze(-1)
        output_x[resolution] = values * visible.to(values.dtype)
        output_m[resolution] = mask * visible.squeeze(-1).to(mask.dtype)
    return output_x, output_m


def piecewise_warp(times: Tensor, eta: Tensor | float, observation_seconds: float = 60.0) -> Tensor:
    """Monotone warp with anchors ``(0,0), (T/2,(0.5+eta)T), (T,T)``."""
    eta_t = torch.as_tensor(eta, dtype=times.dtype, device=times.device)
    midpoint = (0.5 + eta_t) * observation_seconds
    working_times = times
    if eta_t.ndim > 0 and times.ndim == 1:
        working_times = times.unsqueeze(0)
        midpoint = midpoint.unsqueeze(-1)
    left = working_times * midpoint / (observation_seconds / 2.0)
    right = midpoint + (working_times - observation_seconds / 2.0) * (observation_seconds - midpoint) / (observation_seconds / 2.0)
    return torch.where(working_times <= observation_seconds / 2.0, left, right).clamp(0.0, observation_seconds)


def token_centers(tokens: int, observation_seconds: float, device: torch.device, dtype: torch.dtype) -> Tensor:
    return (torch.arange(tokens, device=device, dtype=dtype) + 0.5) * observation_seconds / tokens


def linear_interpolate_tokens(values: Tensor, source_times: Tensor, observation_seconds: float = 60.0) -> Tensor:
    """Interpolate ``[B,M,C]`` values at ``[M]`` or ``[B,M]`` source times."""
    if values.ndim not in (2, 3):
        raise ValueError("values must be [B,M] or [B,M,C]")
    batch, tokens = values.shape[:2]
    centers = token_centers(tokens, observation_seconds, values.device, values.dtype)
    step = observation_seconds / tokens
    fractional = (source_times - centers[0]) / step
    if fractional.ndim == 1:
        fractional = fractional.unsqueeze(0).expand(batch, -1)
    lower = fractional.floor().long().clamp(0, tokens - 1)
    upper = (lower + 1).clamp(0, tokens - 1)
    weight = (fractional - lower.to(fractional.dtype)).clamp(0.0, 1.0)
    if values.ndim == 3:
        channels = values.shape[-1]
        lo = values.gather(1, lower.unsqueeze(-1).expand(-1, -1, channels))
        hi = values.gather(1, upper.unsqueeze(-1).expand(-1, -1, channels))
        return lo + weight.unsqueeze(-1) * (hi - lo)
    lo = values.gather(1, lower)
    hi = values.gather(1, upper)
    return lo + weight * (hi - lo)


def warp_tokens(
    tokens: Tensor,
    reliability: Tensor,
    eta: Tensor | float,
    observation_seconds: float = 60.0,
) -> tuple[Tensor, Tensor, Tensor]:
    centers = token_centers(tokens.shape[1], observation_seconds, tokens.device, tokens.dtype)
    warped_coordinates = piecewise_warp(centers, eta, observation_seconds)
    return (
        linear_interpolate_tokens(tokens, warped_coordinates, observation_seconds),
        linear_interpolate_tokens(reliability, warped_coordinates, observation_seconds),
        warped_coordinates,
    )


def alignment_target(
    source_reliability: Tensor,
    transformed_reliability: Tensor,
    warped_coordinates: Tensor,
    sigma_seconds: float = 2.0,
    observation_seconds: float = 60.0,
    epsilon: float = 1e-8,
) -> tuple[Tensor, Tensor]:
    """Construct Eq. (16), returning ``G[B,M,M]`` and valid columns ``[B,M]``."""
    batch, tokens = source_reliability.shape
    centers = token_centers(tokens, observation_seconds, source_reliability.device, source_reliability.dtype)
    if warped_coordinates.ndim == 1:
        warped_coordinates = warped_coordinates.unsqueeze(0).expand(batch, -1)
    distance = centers.view(1, tokens, 1) - warped_coordinates.view(batch, 1, tokens)
    target = torch.exp(-(distance.square()) / (2.0 * sigma_seconds**2))
    target = target * source_reliability.unsqueeze(-1) * transformed_reliability.unsqueeze(1)
    mass = target.sum(dim=1, keepdim=True)
    valid = (transformed_reliability > 0) & (mass.squeeze(1) > epsilon)
    target = target / mass.clamp_min(epsilon)
    target = target * valid.unsqueeze(1).to(target.dtype)
    return target, valid
