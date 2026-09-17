import torch

from vstamp.models.amp import AlignedMultiscalePooling


def test_amp_shapes_normalization_and_masking() -> None:
    torch.manual_seed(0)
    model = AlignedMultiscalePooling(tokens=30, dropout=0.0)
    features = {
        100: torch.randn(2, 600, 5),
        500: torch.randn(2, 120, 5),
        2000: torch.randn(2, 30, 5),
    }
    masks = {resolution: torch.ones(2, values.shape[1]) for resolution, values in features.items()}
    masks[100][1, 300:] = 0
    masks[500][1, 60:] = 0
    masks[2000][1, 15:] = 0
    z, tokens, reliability, weights = model(features, masks)
    assert z.shape == (2, 512)
    assert tokens.shape == (2, 30, 256)
    assert reliability.shape == (2, 30)
    assert weights.shape == (2, 30, 3)
    torch.testing.assert_close(z.norm(dim=-1), torch.ones(2), atol=1e-5, rtol=1e-5)
    expected_weight_sum = (reliability > 0).to(weights.dtype)
    torch.testing.assert_close(weights.sum(dim=-1), expected_weight_sum, atol=1e-5, rtol=1e-5)
    assert torch.all(reliability >= 0) and torch.all(reliability <= 1)
