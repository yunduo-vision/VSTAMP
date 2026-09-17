from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch import Tensor

from .cache import load_feature_cache
from .common import SessionRecord, load_trace, read_records
from .packet_features import NormalizationStats, aggregate_multiscale


class ProcessedDataset:
    def __init__(self, root: str | Path, split: str | None = None, cache_size: int = 256) -> None:
        self.root = Path(root).expanduser().resolve()
        index = self.root / "sessions.csv"
        if not index.is_file():
            raise FileNotFoundError(
                f"Processed session index not found: {index}. Run the dataset preparation script first."
            )
        records = read_records(index)
        self.records = [record for record in records if split is None or record.split == split]
        if not self.records:
            raise ValueError(f"No sessions available for split={split!r}")
        self.by_id = {record.session_id: record for record in self.records}
        self.stats = NormalizationStats.load(self.root / "normalization_stats.npz")
        self.cache_size = cache_size
        self._cache: OrderedDict[str, tuple[dict[int, np.ndarray], dict[int, np.ndarray]]] = OrderedDict()

    def _load(self, session_id: str) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray]]:
        if session_id in self._cache:
            value = self._cache.pop(session_id)
            self._cache[session_id] = value
            return value
        record = self.by_id[session_id]
        value = load_feature_cache(record.feature_path)
        self._cache[session_id] = value
        while len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)
        return value

    def tensors(
        self,
        session_ids: Iterable[str],
        device: torch.device | str,
        retained_fraction: float | None = None,
    ) -> tuple[dict[int, Tensor], dict[int, Tensor]]:
        ids = list(session_ids)
        if retained_fraction is None or retained_fraction >= 1.0:
            loaded = [self._load(session_id) for session_id in ids]
        else:
            loaded = [self._raw_partial(self.by_id[session_id], retained_fraction) for session_id in ids]
        resolutions = sorted(loaded[0][0])
        features = {
            resolution: torch.from_numpy(np.stack([item[0][resolution] for item in loaded])).to(device)
            for resolution in resolutions
        }
        masks = {
            resolution: torch.from_numpy(np.stack([item[1][resolution] for item in loaded])).to(device)
            for resolution in resolutions
        }
        return features, masks

    def _raw_partial(
        self, record: SessionRecord, retained_fraction: float
    ) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray]]:
        on_wire = "ydms" not in str(self.root).lower()
        trace = load_trace(record, on_wire=on_wire)
        features, masks = aggregate_multiscale(trace)
        features = self.stats.transform(features)
        for resolution in features:
            length = len(masks[resolution])
            centers = (np.arange(length) + 0.5) * 60.0 / length
            visible = centers <= retained_fraction * 60.0
            masks[resolution] = visible.astype(np.float32)
            features[resolution][~visible] = 0.0
        return features, masks

