from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from .common import SessionRecord


@dataclass(frozen=True)
class Episode:
    class_ids: list[str]
    support_ids: list[list[str]]
    query_ids: list[list[str]]
    support_condition: str = ""
    query_condition: str = ""


class EpisodeSampler:
    def __init__(self, records: Iterable[SessionRecord], seed: int) -> None:
        self.records = list(records)
        self.rng = np.random.default_rng(seed)
        self.by_identity: dict[str, list[SessionRecord]] = defaultdict(list)
        for record in self.records:
            self.by_identity[record.video_id].append(record)

    def sample(
        self,
        n_way: int,
        n_shot: int,
        n_query: int,
        support_field: str | None = None,
        support_value: str | None = None,
        query_value: str | None = None,
    ) -> Episode:
        eligible: list[tuple[str, list[SessionRecord], list[SessionRecord]]] = []
        for identity, rows in self.by_identity.items():
            if support_field:
                support_pool = [row for row in rows if getattr(row, support_field) == support_value]
                query_pool = [row for row in rows if getattr(row, support_field) == query_value]
            else:
                support_pool = rows
                query_pool = rows
            if support_value == query_value or not support_field:
                if len(rows) >= n_shot + n_query:
                    eligible.append((identity, support_pool, query_pool))
            elif len(support_pool) >= n_shot and len(query_pool) >= n_query:
                eligible.append((identity, support_pool, query_pool))
        if len(eligible) < n_way:
            raise ValueError(f"Need {n_way} eligible identities, found {len(eligible)}")
        chosen = self.rng.choice(len(eligible), size=n_way, replace=False)
        class_ids: list[str] = []
        support_ids: list[list[str]] = []
        query_ids: list[list[str]] = []
        for index in chosen:
            identity, support_pool, query_pool = eligible[int(index)]
            class_ids.append(identity)
            if support_pool is query_pool or support_value == query_value or not support_field:
                selected = self.rng.choice(len(support_pool), size=n_shot + n_query, replace=False)
                support_ids.append([support_pool[int(i)].session_id for i in selected[:n_shot]])
                query_ids.append([support_pool[int(i)].session_id for i in selected[n_shot:]])
            else:
                support_sel = self.rng.choice(len(support_pool), size=n_shot, replace=False)
                query_sel = self.rng.choice(len(query_pool), size=n_query, replace=False)
                support_ids.append([support_pool[int(i)].session_id for i in support_sel])
                query_ids.append([query_pool[int(i)].session_id for i in query_sel])
        support_flat = {sid for group in support_ids for sid in group}
        query_flat = {sid for group in query_ids for sid in group}
        if support_flat & query_flat:
            raise RuntimeError("Support/query session overlap detected")
        return Episode(class_ids, support_ids, query_ids, support_value or "", query_value or "")


def save_episodes(path: str | Path, episodes: Sequence[Episode]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as stream:
        json.dump([asdict(episode) for episode in episodes], stream, indent=2)


def load_episodes(path: str | Path) -> list[Episode]:
    with Path(path).open("r", encoding="utf-8") as stream:
        return [Episode(**item) for item in json.load(stream)]

