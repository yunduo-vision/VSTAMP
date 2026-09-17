import torch

from vstamp.models.matching import aggregate_candidates, compatibility_responsibilities


def test_responsibilities_sum_to_one_and_log_domain_score_is_finite() -> None:
    overlap = torch.tensor([[[0.9, 0.2], [0.4, 0.6]]])
    similarity = torch.tensor([[[0.8, 0.7], [0.1, 0.5]]])
    pair_scores = torch.tensor([[[100.0, 90.0], [80.0, 70.0]]])
    weights = compatibility_responsibilities(overlap, similarity)
    torch.testing.assert_close(weights.sum(dim=-1), torch.ones(1, 2))
    candidate, returned = aggregate_candidates(pair_scores, overlap, similarity)
    assert candidate.shape == (1, 2)
    assert torch.isfinite(candidate).all()
    torch.testing.assert_close(returned.sum(dim=-1), torch.ones(1, 2))

