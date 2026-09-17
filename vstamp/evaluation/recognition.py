from __future__ import annotations

from dataclasses import asdict
from typing import Iterable

import torch

from ..data.dataset import ProcessedDataset
from ..data.episode_sampler import Episode
from ..models.vstamp import Fingerprint, VSTAMP


def _reshape_fingerprint(fingerprint: Fingerprint, ways: int, shots: int) -> Fingerprint:
    return Fingerprint(
        fingerprint.z.reshape(ways, shots, -1),
        fingerprint.tokens.reshape(ways, shots, *fingerprint.tokens.shape[1:]),
        fingerprint.reliability.reshape(ways, shots, -1),
        None,
    )


def evaluate_episodes(
    model: VSTAMP,
    dataset: ProcessedDataset,
    episodes: Iterable[Episode],
    device: torch.device,
    retained_fraction: float | None = None,
) -> tuple[dict[str, float], list[dict[str, object]]]:
    model.eval()
    correct = 0
    total = 0
    rows: list[dict[str, object]] = []
    with torch.no_grad():
        for episode_index, episode in enumerate(episodes):
            support_ids = [sid for group in episode.support_ids for sid in group]
            query_ids = [sid for group in episode.query_ids for sid in group]
            ways = len(episode.class_ids)
            shots = len(episode.support_ids[0])
            support_x, support_m = dataset.tensors(support_ids, device)
            query_x, query_m = dataset.tensors(query_ids, device, retained_fraction=retained_fraction)
            support = _reshape_fingerprint(model.encode(support_x, support_m), ways, shots)
            query = model.encode(query_x, query_m)
            scores, responsibilities, _ = model.score_candidates(query, support)
            predictions = scores.argmax(dim=-1)
            queries_per_class = len(episode.query_ids[0])
            targets = torch.arange(ways, device=device).repeat_interleave(queries_per_class)
            correct += int((predictions == targets).sum())
            total += len(targets)
            for index, session_id in enumerate(query_ids):
                rows.append(
                    {
                        "episode": episode_index,
                        "session_id": session_id,
                        "target": episode.class_ids[int(targets[index])],
                        "prediction": episode.class_ids[int(predictions[index])],
                        "correct": int(predictions[index] == targets[index]),
                        "max_score": float(scores[index].max().cpu()),
                        "responsibility_max": float(responsibilities[index].max().cpu()),
                    }
                )
    return {"accuracy": correct / max(total, 1), "correct": correct, "queries": total}, rows

