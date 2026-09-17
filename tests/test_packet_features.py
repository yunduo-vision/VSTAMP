import numpy as np

from vstamp.data.common import PacketTrace
from vstamp.data.packet_features import NormalizationStats, aggregate_multiscale


def test_multiscale_shapes_and_feature_semantics() -> None:
    trace = PacketTrace(
        timestamps=np.asarray([10.0, 10.05, 10.15, 69.99, 70.0]),
        directions=np.asarray([1, -1, 1, -1, 1], dtype=np.int8),
        lengths=np.asarray([100, 50, 300, 200, 999], dtype=np.float32),
    )
    features, masks = aggregate_multiscale(trace)
    assert features[100].shape == (600, 5)
    assert features[500].shape == (120, 5)
    assert features[2000].shape == (30, 5)
    assert all(np.all(mask == 1) for mask in masks.values())
    np.testing.assert_allclose(features[100][0, 0], np.log1p(100))
    np.testing.assert_allclose(features[100][0, 1], np.log1p(50))
    np.testing.assert_allclose(features[100][0, 4], np.log1p(100))
    assert np.isclose(features[100][:, 0].max(), np.log1p(300))


def test_normalization_fits_training_features_only() -> None:
    first = {100: np.zeros((2, 5)), 500: np.zeros((2, 5)), 2000: np.zeros((2, 5))}
    second = {key: np.ones((2, 5)) for key in first}
    stats = NormalizationStats.fit([first, second], first)
    transformed = stats.transform({key: np.full((1, 5), 0.5) for key in first})
    assert all(np.allclose(value, 0.0) for value in transformed.values())

