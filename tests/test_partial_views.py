import torch

from vstamp.data.partial_views import (
    alignment_target,
    piecewise_warp,
    prefix_retention,
    token_centers,
    warp_tokens,
)


def test_prefix_is_shared_across_resolutions() -> None:
    features = {
        100: torch.ones(2, 600, 5),
        500: torch.ones(2, 120, 5),
        2000: torch.ones(2, 30, 5),
    }
    masks = {resolution: torch.ones(2, values.shape[1]) for resolution, values in features.items()}
    _, output_masks = prefix_retention(features, masks, torch.tensor([0.5, 1.0]))
    assert output_masks[100][0].sum() == 300
    assert output_masks[500][0].sum() == 60
    assert output_masks[2000][0].sum() == 15
    assert output_masks[100][1].sum() == 600


def test_warp_endpoints_monotonicity_and_alignment_target() -> None:
    times = torch.linspace(0, 60, 101)
    for eta in (-0.3, 0.0, 0.3):
        warped = piecewise_warp(times, eta)
        assert torch.isclose(warped[0], torch.tensor(0.0))
        assert torch.isclose(warped[-1], torch.tensor(60.0))
        assert torch.all(warped[1:] >= warped[:-1])
    tokens = torch.randn(2, 30, 256)
    reliability = torch.ones(2, 30)
    transformed, transformed_r, coordinates = warp_tokens(tokens, reliability, torch.tensor([-0.2, 0.2]))
    target, valid = alignment_target(reliability, transformed_r, coordinates)
    assert transformed.shape == tokens.shape
    assert target.shape == (2, 30, 30)
    sums = target.sum(dim=1)
    torch.testing.assert_close(sums[valid], torch.ones_like(sums[valid]), atol=1e-5, rtol=1e-5)

