from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.neighbors import NearestNeighbors


def dense(x: Any) -> np.ndarray:
    """Return a dense numpy array from a dense or sparse matrix."""
    return x.toarray() if sparse.issparse(x) else np.asarray(x)


def read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: str | Path, value: dict[str, Any] | list[Any]) -> None:
    Path(path).write_text(json.dumps(value, indent=2), encoding="utf-8")


def normalized_entropy(prob: np.ndarray) -> np.ndarray:
    p = np.clip(prob, 1e-12, 1.0)
    p = p / np.maximum(p.sum(axis=1, keepdims=True), 1e-12)
    if p.shape[1] <= 1:
        return np.zeros(p.shape[0], dtype=float)
    return -(p * np.log2(p)).sum(axis=1) / np.log2(p.shape[1])


def scale_robust(values: np.ndarray, reference: np.ndarray | None = None) -> np.ndarray:
    ref = values if reference is None else reference
    lo = float(np.quantile(ref, 0.05))
    hi = float(np.quantile(ref, 0.95))
    if hi <= lo:
        return np.zeros_like(values, dtype=float)
    return np.clip((values - lo) / (hi - lo), 0.0, 1.0)


def ordered_labels(values: pd.Series) -> list[str]:
    labels = values.dropna().astype(str)
    labels = labels[~labels.str.lower().isin(["", "nan", "none"])]
    return sorted(labels.unique().tolist(), key=lambda value: value.lower())


def one_hot(labels: pd.Series, known_labels: list[str]) -> np.ndarray:
    label_to_idx = {label: idx for idx, label in enumerate(known_labels)}
    y = np.zeros((len(labels), len(known_labels)), dtype=np.float32)
    idx = labels.astype(str).map(label_to_idx).to_numpy()
    valid = pd.notna(idx)
    y[np.where(valid)[0], idx[valid].astype(int)] = 1.0
    return y


def nearest_neighbor_novelty(x_scaled: np.ndarray, x_train_scaled: np.ndarray) -> np.ndarray:
    if x_train_scaled.shape[0] < 2:
        return np.zeros(x_scaled.shape[0], dtype=float)
    nbrs = NearestNeighbors(n_neighbors=min(10, x_train_scaled.shape[0]), metric="euclidean")
    nbrs.fit(x_train_scaled)
    train_dist = nbrs.kneighbors(x_train_scaled, return_distance=True)[0][:, -1]
    all_dist = nbrs.kneighbors(x_scaled, return_distance=True)[0][:, -1]
    return scale_robust(all_dist, train_dist)


def centroid_reference_novelty(
    x_query_scaled: np.ndarray,
    x_reference_scaled: np.ndarray,
    reference_labels: pd.Series,
    known_labels: list[str],
    *,
    chunk_size: int = 10000,
) -> np.ndarray:
    label_values = reference_labels.astype(str).to_numpy()
    best = np.full(x_query_scaled.shape[0], np.inf, dtype=np.float32)
    ref_scores = []
    for label in known_labels:
        mask = label_values == label
        if mask.sum() < 2:
            continue
        ref = x_reference_scaled[mask]
        centroid = ref.mean(axis=0, keepdims=True)
        ref_dist = np.sqrt(np.mean((ref - centroid) ** 2, axis=1))
        radius = float(np.quantile(ref_dist, 0.95))
        if radius < 1e-6:
            radius = 1.0
        ref_scores.append(ref_dist / radius)
        for start in range(0, x_query_scaled.shape[0], chunk_size):
            stop = min(start + chunk_size, x_query_scaled.shape[0])
            dist = np.sqrt(np.mean((x_query_scaled[start:stop] - centroid) ** 2, axis=1)) / radius
            best[start:stop] = np.minimum(best[start:stop], dist.astype(np.float32))
    if not ref_scores:
        return np.zeros(x_query_scaled.shape[0], dtype=float)
    best[~np.isfinite(best)] = float(np.nanmax(best[np.isfinite(best)])) if np.isfinite(best).any() else 0.0
    return scale_robust(best.astype(float), np.concatenate(ref_scores).astype(float))

