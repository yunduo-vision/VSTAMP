from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np

from .common import PacketTrace


@dataclass
class NormalizationStats:
    means: dict[int, np.ndarray]
    stds: dict[int, np.ndarray]
    epsilon: float = 1e-8

    @classmethod
    def fit(
        cls,
        sessions: Iterable[Mapping[int, np.ndarray]],
        resolutions_ms: Iterable[int],
        epsilon: float = 1e-8,
    ) -> "NormalizationStats":
        values: dict[int, list[np.ndarray]] = {int(r): [] for r in resolutions_ms}
        for session in sessions:
            for resolution in values:
                values[resolution].append(np.asarray(session[resolution], dtype=np.float64))
        means: dict[int, np.ndarray] = {}
        stds: dict[int, np.ndarray] = {}
        for resolution, arrays in values.items():
            if not arrays:
                raise ValueError(f"No training features available at {resolution} ms")
            joined = np.concatenate(arrays, axis=0)
            means[resolution] = joined.mean(axis=0).astype(np.float32)
            stds[resolution] = np.maximum(joined.std(axis=0), epsilon).astype(np.float32)
        return cls(means, stds, epsilon)

    def transform(self, features: Mapping[int, np.ndarray]) -> dict[int, np.ndarray]:
        return {
            resolution: ((np.asarray(array) - self.means[resolution]) / self.stds[resolution]).astype(np.float32)
            for resolution, array in features.items()
        }

    def save(self, path: str | Path) -> None:
        payload: dict[str, np.ndarray] = {"epsilon": np.asarray(self.epsilon)}
        for resolution in self.means:
            payload[f"mean_{resolution}"] = self.means[resolution]
            payload[f"std_{resolution}"] = self.stds[resolution]
        np.savez_compressed(path, **payload)

    @classmethod
    def load(cls, path: str | Path) -> "NormalizationStats":
        with np.load(path, allow_pickle=False) as data:
            resolutions = sorted(int(key.split("_")[1]) for key in data.files if key.startswith("mean_"))
            return cls(
                {r: data[f"mean_{r}"].astype(np.float32) for r in resolutions},
                {r: data[f"std_{r}"].astype(np.float32) for r in resolutions},
                float(data["epsilon"]),
            )


def aggregate_trace(trace: PacketTrace, resolution_ms: int, observation_seconds: float = 60.0) -> np.ndarray:
    """Aggregate one trace into Eq. (3), returning ``[L_delta, 5]`` log-features."""
    bins = int(round(observation_seconds * 1000.0 / resolution_ms))
    output = np.zeros((bins, 5), dtype=np.float64)
    if len(trace.timestamps) == 0:
        return output.astype(np.float32)
    relative = trace.timestamps - trace.timestamps[0]
    keep = (relative >= 0.0) & (relative < observation_seconds)
    relative = relative[keep]
    directions = trace.directions[keep]
    lengths = trace.lengths[keep].astype(np.float64)
    indices = np.floor(relative * 1000.0 / resolution_ms).astype(np.int64)
    down = directions == 1
    up = directions == -1
    np.add.at(output[:, 0], indices[down], lengths[down])
    np.add.at(output[:, 1], indices[up], lengths[up])
    np.add.at(output[:, 2], indices[down], 1.0)
    np.add.at(output[:, 3], indices[up], 1.0)
    down_lengths = np.zeros(bins, dtype=np.float64)
    np.add.at(down_lengths, indices[down], lengths[down])
    nonempty = output[:, 2] > 0
    output[nonempty, 4] = down_lengths[nonempty] / output[nonempty, 2]
    return np.log1p(output).astype(np.float32)


def aggregate_multiscale(
    trace: PacketTrace,
    resolutions_ms: Iterable[int] = (100, 500, 2000),
    observation_seconds: float = 60.0,
) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray]]:
    features = {
        int(resolution): aggregate_trace(trace, int(resolution), observation_seconds)
        for resolution in resolutions_ms
    }
    masks = {resolution: np.ones(array.shape[0], dtype=np.float32) for resolution, array in features.items()}
    return features, masks

