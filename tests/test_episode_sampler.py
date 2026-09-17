from vstamp.data.common import SessionRecord
from vstamp.data.episode_sampler import EpisodeSampler


def test_episode_counts_disjointness_and_seed_reproducibility() -> None:
    records = [
        SessionRecord(session_id=f"v{video}-s{session}", video_id=f"v{video}", trace_path="unused")
        for video in range(12)
        for session in range(8)
    ]
    first = EpisodeSampler(records, seed=7).sample(10, 2, 4)
    second = EpisodeSampler(records, seed=7).sample(10, 2, 4)
    assert first == second
    assert len(first.class_ids) == 10
    assert all(len(group) == 2 for group in first.support_ids)
    assert all(len(group) == 4 for group in first.query_ids)
    support = {session for group in first.support_ids for session in group}
    query = {session for group in first.query_ids for session in group}
    assert support.isdisjoint(query)

