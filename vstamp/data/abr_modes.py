from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


QUALITY_ALIASES = (
    "bitrate",
    "selected_bitrate",
    "requested_bitrate",
    "quality",
    "quality_index",
    "resolution",
    "video_quality",
)
TIME_ALIASES = ("timestamp", "time", "relative_timestamp", "playback_time", "t")
MODE_NAMES = ("low_or_late", "transitional", "high_stable")


def quality_trajectory(path: str | Path, intervals: int = 30, seconds: float = 60.0) -> np.ndarray:
    frame = pd.read_csv(path)
    lower = {str(column).lower(): str(column) for column in frame.columns}
    q_col = next((lower[name] for name in QUALITY_ALIASES if name in lower), None)
    t_col = next((lower[name] for name in TIME_ALIASES if name in lower), None)
    if q_col is None or t_col is None:
        raise ValueError(f"{path} must contain a time column and a quality/bitrate column")
    times = pd.to_numeric(frame[t_col], errors="coerce").to_numpy(dtype=np.float64)
    quality = pd.to_numeric(frame[q_col], errors="coerce").to_numpy(dtype=np.float64)
    keep = np.isfinite(times) & np.isfinite(quality)
    times, quality = times[keep], quality[keep]
    if times.size == 0:
        raise ValueError(f"No numeric quality trajectory values in {path}")
    times = times - times.min()
    centers = (np.arange(intervals, dtype=np.float64) + 0.5) * seconds / intervals
    order = np.argsort(times, kind="stable")
    times, quality = times[order], quality[order]
    indices = np.searchsorted(times, centers, side="right") - 1
    indices = np.clip(indices, 0, len(quality) - 1)
    return quality[indices].astype(np.float32)


@dataclass
class ABRModeModel:
    scaler: StandardScaler
    kmeans: KMeans
    label_map: dict[int, str]

    def predict(self, trajectories: np.ndarray) -> list[str]:
        transformed = self.scaler.transform(np.asarray(trajectories, dtype=np.float64))
        labels = self.kmeans.predict(transformed)
        return [self.label_map[int(label)] for label in labels]

    def save(self, path: str | Path) -> None:
        with Path(path).open("wb") as stream:
            pickle.dump(self, stream)

    @classmethod
    def load(cls, path: str | Path) -> "ABRModeModel":
        with Path(path).open("rb") as stream:
            model = pickle.load(stream)
        if not isinstance(model, cls):
            raise TypeError("Invalid ABR mode model")
        return model


def fit_abr_modes(
    trajectories: Iterable[np.ndarray],
    k_values: Iterable[int] = range(2, 11),
    seed: int = 0,
) -> tuple[ABRModeModel, dict[int, float]]:
    data = np.asarray(list(trajectories), dtype=np.float64)
    if data.ndim != 2 or data.shape[0] < 3:
        raise ValueError("At least three quality trajectories are required for ABR clustering")
    scaler = StandardScaler().fit(data)
    normalized = scaler.transform(data)
    scores: dict[int, float] = {}
    models: dict[int, KMeans] = {}
    for k in k_values:
        if k >= len(normalized):
            continue
        model = KMeans(n_clusters=int(k), random_state=seed, n_init=20).fit(normalized)
        scores[int(k)] = float(silhouette_score(normalized, model.labels_))
        models[int(k)] = model
    if not scores:
        raise ValueError("Not enough trajectories for the requested K range")
    best_k = max(scores, key=scores.get)
    kmeans = models[best_k]
    centroid_quality = scaler.inverse_transform(kmeans.cluster_centers_).mean(axis=1)
    ordered = np.argsort(centroid_quality)
    if best_k == 3:
        label_map = {int(cluster): MODE_NAMES[rank] for rank, cluster in enumerate(ordered)}
    else:
        label_map = {int(cluster): f"mode_{rank}" for rank, cluster in enumerate(ordered)}
    return ABRModeModel(scaler, kmeans, label_map), scores

