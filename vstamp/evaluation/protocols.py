from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np

from ..data.common import SessionRecord
from ..data.episode_sampler import Episode, EpisodeSampler, load_episodes, save_episodes


LONG_MODES = ("low_or_late", "transitional", "high_stable")
YDMS_MODES = ("starved", "transitional", "high_stable")
BANDWIDTH_PAIRS = (("bw1", "bw8"), ("bw8", "bw1"), ("bw1", "bw2"), ("bw2", "bw1"))


def _balanced_support_episode(records: list[SessionRecord], rng: np.random.Generator, n_query: int = 2) -> Episode:
    grouped: dict[str, list[SessionRecord]] = defaultdict(list)
    for record in records:
        grouped[record.video_id].append(record)
    eligible = []
    for identity, rows in grouped.items():
        by_mode = {mode: [row for row in rows if row.abr_mode == mode] for mode in LONG_MODES}
        if all(by_mode[mode] for mode in LONG_MODES) and len(rows) >= 3 + n_query:
            eligible.append((identity, rows, by_mode))
    if len(eligible) < 5:
        raise ValueError("Balanced 3-shot evaluation needs five identities with all three ABR modes")
    selected = rng.choice(len(eligible), size=5, replace=False)
    classes: list[str] = []
    support_ids: list[list[str]] = []
    query_ids: list[list[str]] = []
    for index in selected:
        identity, rows, by_mode = eligible[int(index)]
        support = [by_mode[mode][int(rng.integers(len(by_mode[mode])))] for mode in LONG_MODES]
        support_set = {row.session_id for row in support}
        query_pool = [row for row in rows if row.session_id not in support_set]
        query = rng.choice(len(query_pool), size=n_query, replace=False)
        classes.append(identity)
        support_ids.append([row.session_id for row in support])
        query_ids.append([query_pool[int(i)].session_id for i in query])
    return Episode(classes, support_ids, query_ids)


def generate_episodes(
    records: list[SessionRecord],
    protocol: str,
    count: int,
    seed: int,
    dataset_name: str,
) -> list[Episode]:
    sampler = EpisodeSampler(records, seed)
    rng = np.random.default_rng(seed)
    episodes: list[Episode] = []
    for index in range(count):
        if protocol == "cross_mode":
            modes = LONG_MODES if dataset_name == "longenough" else YDMS_MODES[1:]
            pairs = [(left, right) for left in modes for right in modes if left != right]
            left, right = pairs[index % len(pairs)]
            ways, shots, queries = (5, 2, 2) if dataset_name == "longenough" else (6, 5, 4)
            episodes.append(sampler.sample(ways, shots, queries, "abr_mode", left, right))
        elif protocol == "same_mode":
            mode = LONG_MODES[index % len(LONG_MODES)]
            episodes.append(sampler.sample(5, 2, 2, "abr_mode", mode, mode))
        elif protocol == "balanced_3shot":
            episodes.append(_balanced_support_episode(records, rng))
        elif protocol == "random_5shot":
            ways = 10 if dataset_name == "ydms" else 5
            episodes.append(sampler.sample(ways, 5, 4))
        elif protocol == "cross_bandwidth":
            left, right = BANDWIDTH_PAIRS[index % len(BANDWIDTH_PAIRS)]
            episodes.append(sampler.sample(5, 2, 2, "bandwidth", left, right))
        elif protocol.startswith("bandwidth_"):
            _, left, right = protocol.split("_", maxsplit=2)
            episodes.append(sampler.sample(5, 2, 2, "bandwidth", left, right))
        elif protocol.startswith("unseen_"):
            bandwidth = protocol.removeprefix("unseen_")
            episodes.append(sampler.sample(5, 2, 2, "bandwidth", bandwidth, bandwidth))
        else:
            raise ValueError(f"Unsupported protocol: {protocol}")
    return episodes


def fixed_episodes(
    records: list[SessionRecord],
    protocol: str,
    count: int,
    seed: int,
    dataset_name: str,
    cache_dir: str | Path,
) -> list[Episode]:
    path = Path(cache_dir) / f"{protocol}_seed_{seed}_{count}.json"
    if path.is_file():
        return load_episodes(path)
    episodes = generate_episodes(records, protocol, count, seed, dataset_name)
    save_episodes(path, episodes)
    return episodes

