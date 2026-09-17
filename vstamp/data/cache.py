from __future__ import annotations

from pathlib import Path
from typing import Mapping

import numpy as np


def save_feature_cache(
    path: str | Path,
    features: Mapping[int, np.ndarray],
    masks: Mapping[int, np.ndarray],
) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, np.ndarray] = {}
    for resolution, array in features.items():
        payload[f"x_{resolution}"] = np.asarray(array, dtype=np.float32)
        payload[f"m_{resolution}"] = np.asarray(masks[resolution], dtype=np.float32)
    np.savez_compressed(output, **payload)


def load_feature_cache(path: str | Path) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray]]:
    with np.load(path, allow_pickle=False) as data:
        resolutions = sorted(int(key.split("_")[1]) for key in data.files if key.startswith("x_"))
        features = {r: data[f"x_{r}"].astype(np.float32) for r in resolutions}
        masks = {r: data[f"m_{r}"].astype(np.float32) for r in resolutions}
    return features, masks

