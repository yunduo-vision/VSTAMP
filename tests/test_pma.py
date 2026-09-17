import torch

from vstamp.models.pma import PartialMonotoneAlignment, pma_autograd_reference, pma_explicit


def test_pma_shape_finite_and_reference_agreement() -> None:
    torch.manual_seed(1)
    affinity = torch.randn(3, 7, 6, dtype=torch.float64, requires_grad=True)
    valid = torch.ones_like(affinity, dtype=torch.bool)
    score, mass = pma_explicit(affinity, valid, gamma=0.2, gap=0.1)
    reference_score, reference_mass = pma_autograd_reference(affinity, valid, gamma=0.2, gap=0.1)
    assert score.shape == (3,)
    assert mass.shape == (3, 7, 6)
    assert torch.isfinite(score).all() and torch.isfinite(mass).all()
    assert torch.all(mass >= 0)
    torch.testing.assert_close(score, reference_score, atol=1e-10, rtol=1e-10)
    assert float((mass - reference_mass).abs().max()) < 1e-5


def test_identity_and_shifted_correspondence() -> None:
    size = 12
    indices = torch.arange(size)
    diagonal_affinity = -4.0 * torch.ones(1, size, size)
    diagonal_affinity[0, indices, indices] = 4.0
    _, diagonal_mass = pma_explicit(
        diagonal_affinity, torch.ones_like(diagonal_affinity, dtype=torch.bool), gamma=0.05, gap=0.1
    )
    diag = diagonal_mass[0].diagonal().sum()
    assert diag / diagonal_mass.sum() > 0.8

    shifted_affinity = -4.0 * torch.ones(1, size, size)
    shifted_affinity[0, indices[:-2], indices[2:]] = 4.0
    _, shifted_mass = pma_explicit(
        shifted_affinity, torch.ones_like(shifted_affinity, dtype=torch.bool), gamma=0.05, gap=0.1
    )
    ridge = shifted_mass[0, indices[:-2], indices[2:]].sum()
    assert ridge / shifted_mass.sum() > 0.7


def test_partial_validity_and_second_order_gradient() -> None:
    torch.manual_seed(3)
    affinity = torch.randn(2, 8, 8, requires_grad=True)
    valid = torch.ones_like(affinity, dtype=torch.bool)
    valid[:, :2, :] = False
    valid[:, :, -2:] = False
    score, mass = pma_explicit(affinity, valid)
    assert torch.count_nonzero(mass.masked_select(~valid)) == 0
    loss = score.mean() + mass.square().mean()
    loss.backward()
    assert affinity.grad is not None
    assert torch.isfinite(affinity.grad).all()


def test_full_alignment_module_parameter_gradients() -> None:
    torch.manual_seed(5)
    module = PartialMonotoneAlignment()
    z_a = torch.nn.functional.normalize(torch.randn(2, 512), dim=-1)
    z_b = torch.nn.functional.normalize(torch.randn(2, 512), dim=-1)
    tokens_a = torch.randn(2, 30, 256, requires_grad=True)
    tokens_b = torch.randn(2, 30, 256, requires_grad=True)
    reliability = torch.ones(2, 30)
    metrics = module(z_a, tokens_a, reliability, z_b, tokens_b, reliability)
    loss = metrics.pair_score.mean() + metrics.overlap.mean()
    loss.backward()
    assert module.local_projection.weight.grad is not None
    assert torch.isfinite(module.local_projection.weight.grad).all()

