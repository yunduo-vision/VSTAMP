from __future__ import annotations

from collections import defaultdict

import numpy as np
import torch

from ..data.dataset import ProcessedDataset
from ..models.vstamp import Fingerprint, VSTAMP
from ..utils.metrics import open_set_metrics


def evaluate_open_set(
    model: VSTAMP,
    dataset: ProcessedDataset,
    partitions: int,
    seed: int,
    device: torch.device,
) -> list[dict[str, float]]:
    grouped: dict[str, list[str]] = defaultdict(list)
    for record in dataset.records:
        grouped[record.video_id].append(record.session_id)
    identities = sorted(grouped)
    if len(identities) != 30:
        raise ValueError(f"Open-set protocol requires 30 LongEnough test identities, found {len(identities)}")
    rng = np.random.default_rng(seed)
    results = []
    model.eval()
    for partition in range(partitions):
        order = rng.permutation(identities)
        enrolled, unknown = order[:15], set(order[15:])
        support_ids: list[str] = []
        known_queries: list[str] = []
        for identity in enrolled:
            rows = grouped[str(identity)]
            if len(rows) < 6:
                raise ValueError(f"Identity {identity} needs at least six sessions for open-set evaluation")
            selected = rng.choice(len(rows), size=5, replace=False)
            support = [rows[int(index)] for index in selected]
            support_set = set(support)
            support_ids.extend(support)
            known_queries.extend([row for row in rows if row not in support_set])
        unknown_queries = [session for identity in unknown for session in grouped[identity]]
        support_x, support_m = dataset.tensors(support_ids, device)
        with torch.no_grad():
            flat = model.encode(support_x, support_m)
            supports = Fingerprint(
                flat.z.reshape(15, 5, -1),
                flat.tokens.reshape(15, 5, *flat.tokens.shape[1:]),
                flat.reliability.reshape(15, 5, -1),
            )
        confidence: list[float] = []
        labels: list[int] = []
        for label, query_ids in ((1, known_queries), (0, unknown_queries)):
            for start in range(0, len(query_ids), 32):
                batch_ids = query_ids[start : start + 32]
                query_x, query_m = dataset.tensors(batch_ids, device)
                with torch.no_grad():
                    scores, _, _ = model.score_candidates(model.encode(query_x, query_m), supports)
                confidence.extend(scores.max(dim=-1).values.cpu().tolist())
                labels.extend([label] * len(batch_ids))
        metrics = open_set_metrics(np.asarray(labels), np.asarray(confidence))
        metrics["partition"] = float(partition)
        results.append(metrics)
    return results
