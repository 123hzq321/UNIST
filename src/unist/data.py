from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

from .utils import centroid_reference_novelty, dense, nearest_neighbor_novelty, one_hot, ordered_labels, read_json


@dataclass
class UnistTask:
    spatial: ad.AnnData
    known_labels: list[str]
    x_scaled: np.ndarray
    x_train_scaled: np.ndarray
    y_train: np.ndarray
    novelty: np.ndarray
    disagreement_reference: str | None
    training_source: str
    n_train_units: int


def has_exact_truth(dataset_dir: Path, metadata: dict) -> bool:
    if str(metadata.get("training_mode", "")).lower() in {"reference_supervised", "scrna_reference"}:
        return False
    truth_path = dataset_dir / "ground_truth_proportions.csv"
    if "known_cell_types" not in metadata or not truth_path.exists():
        return False
    try:
        head = pd.read_csv(truth_path, nrows=2)
    except Exception:
        return False
    return {"spot_id", "cell_type", "true_proportion"}.issubset(head.columns) and not head.empty


def load_pseudo_truth_task(dataset_dir: str | Path) -> UnistTask:
    dataset_dir = Path(dataset_dir)
    spatial = ad.read_h5ad(dataset_dir / "spatial.h5ad")
    metadata = read_json(dataset_dir / "metadata.json")
    known_labels = metadata["known_cell_types"]
    gt = pd.read_csv(dataset_dir / "ground_truth_proportions.csv")
    gt_wide = gt.pivot_table(index="spot_id", columns="cell_type", values="true_proportion", aggfunc="mean").fillna(0)
    for label in known_labels:
        if label not in gt_wide:
            gt_wide[label] = 0.0
    gt_wide = gt_wide.loc[spatial.obs_names.astype(str), known_labels]

    obs = spatial.obs.copy()
    x = np.log1p(dense(spatial.X).astype(np.float32))
    train_mask = obs["split"].astype(str).eq("train").to_numpy()
    if "unknown_fraction" in obs:
        train_mask &= obs["unknown_fraction"].astype(float).to_numpy() <= 1e-8
    if "scenario" in obs:
        train_mask &= obs["scenario"].astype(str).isin(["in_distribution", "ambiguous_known"]).to_numpy()
    if train_mask.sum() < 20:
        raise ValueError("Too few train pseudo-spots for UNIST")

    x_train = x[train_mask]
    mean = x_train.mean(axis=0, keepdims=True)
    std = x_train.std(axis=0, keepdims=True)
    std[std < 1e-6] = 1.0
    x_scaled = (x - mean) / std
    y = gt_wide.to_numpy(dtype=np.float32)
    y = y / np.maximum(y.sum(axis=1, keepdims=True), 1e-12)

    return UnistTask(
        spatial=spatial,
        known_labels=known_labels,
        x_scaled=x_scaled,
        x_train_scaled=x_scaled[train_mask],
        y_train=y[train_mask],
        novelty=nearest_neighbor_novelty(x_scaled, x_scaled[train_mask]),
        disagreement_reference=None,
        training_source="pseudo_spot_ground_truth",
        n_train_units=int(train_mask.sum()),
    )


def load_reference_supervised_task(dataset_dir: str | Path, seed: int = 20260726) -> UnistTask:
    del seed
    dataset_dir = Path(dataset_dir)
    spatial = ad.read_h5ad(dataset_dir / "spatial.h5ad")
    sc_ref = ad.read_h5ad(dataset_dir / "sc_reference.h5ad")
    metadata = read_json(dataset_dir / "metadata.json")
    label_col = "cell_type" if "cell_type" in sc_ref.obs.columns else str(metadata.get("sc_label_col", "cell_type"))
    if label_col not in sc_ref.obs.columns:
        raise KeyError(f"No usable scRNA label column found; tried {label_col!r}")
    sc_labels = sc_ref.obs[label_col].astype(str)
    known_labels = ordered_labels(sc_labels)
    if len(known_labels) < 2:
        raise ValueError("Reference-supervised task needs at least two scRNA cell types")

    x_train = np.log1p(dense(sc_ref.X).astype(np.float32))
    x_query = np.log1p(dense(spatial.X).astype(np.float32))
    mean = x_train.mean(axis=0, keepdims=True)
    std = x_train.std(axis=0, keepdims=True)
    std[std < 1e-6] = 1.0
    x_train_scaled = (x_train - mean) / std
    x_scaled = (x_query - mean) / std
    y_train = one_hot(sc_labels, known_labels)
    train_keep = y_train.sum(axis=1) > 0
    x_train_scaled = x_train_scaled[train_keep]
    y_train = y_train[train_keep]
    novelty = centroid_reference_novelty(x_scaled, x_train_scaled, sc_labels[train_keep], known_labels)

    return UnistTask(
        spatial=spatial,
        known_labels=known_labels,
        x_scaled=x_scaled,
        x_train_scaled=x_train_scaled,
        y_train=y_train,
        novelty=novelty,
        disagreement_reference="query",
        training_source=f"scrna_reference:{label_col}",
        n_train_units=int(x_train_scaled.shape[0]),
    )


def load_task(dataset_dir: str | Path, seed: int = 20260726) -> UnistTask:
    dataset_dir = Path(dataset_dir)
    metadata = read_json(dataset_dir / "metadata.json")
    if has_exact_truth(dataset_dir, metadata):
        return load_pseudo_truth_task(dataset_dir)
    return load_reference_supervised_task(dataset_dir, seed=seed)

