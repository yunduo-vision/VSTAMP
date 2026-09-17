from __future__ import annotations

from typing import Iterable

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve


def mean_sample_std(values: Iterable[float]) -> tuple[float, float]:
    array = np.asarray(list(values), dtype=np.float64)
    if array.size == 0:
        return float("nan"), float("nan")
    return float(array.mean()), float(array.std(ddof=1)) if array.size > 1 else 0.0


def open_set_metrics(labels_known: np.ndarray, confidence: np.ndarray) -> dict[str, float]:
    labels = np.asarray(labels_known, dtype=np.int64)
    scores = np.asarray(confidence, dtype=np.float64)
    auc = roc_auc_score(labels, scores)
    fpr, tpr, _ = roc_curve(labels, scores)
    eligible = np.flatnonzero(tpr >= 0.95)
    fpr95 = float(fpr[eligible[0]]) if eligible.size else 1.0
    return {"auroc": float(auc), "fpr_at_95_tpr": fpr95}

