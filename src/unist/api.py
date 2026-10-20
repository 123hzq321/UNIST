from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .data import UnistTask, load_task
from .models import MLPBackbone, predict_models, train_mlp
from .utils import normalized_entropy, scale_robust, write_json


@dataclass
class UnistConfig:
    model: str = "mlp_ensemble"
    epochs: int = 80
    ensemble_size: int = 5
    hidden: int = 128
    dropout: float = 0.2
    batch_size: int = 128
    predict_batch_size: int = 4096
    lr: float = 1e-3
    device: str = "cpu"
    seed: int = 20260726
    method_name: str | None = None


def _method_name(config: UnistConfig) -> str:
    if config.method_name:
        return config.method_name
    if config.model == "evidential_mlp":
        return "unist_evidential_mlp"
    return "unist_mlp_ensemble"


def _train_and_predict(task: UnistTask, config: UnistConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[list[float]]]:
    evidential = config.model == "evidential_mlp"
    n_models = 1 if evidential else config.ensemble_size
    models = []
    all_losses = []
    for i in range(n_models):
        model = MLPBackbone(
            task.x_scaled.shape[1],
            len(task.known_labels),
            hidden=config.hidden,
            dropout=config.dropout,
            evidential=evidential,
        )
        losses = train_mlp(
            model,
            task.x_train_scaled,
            task.y_train,
            evidential=evidential,
            epochs=config.epochs,
            batch_size=config.batch_size,
            lr=config.lr,
            seed=config.seed + i,
            device=config.device,
        )
        models.append(model)
        all_losses.append(losses)

    mean_prob, disagreement_raw, strength = predict_models(
        models,
        task.x_scaled,
        evidential=evidential,
        device=config.device,
        batch_size=config.predict_batch_size,
    )
    entropy = normalized_entropy(mean_prob)
    if task.disagreement_reference == "query":
        disagreement = scale_robust(disagreement_raw, disagreement_raw)
    else:
        _, train_disagreement_raw, _ = predict_models(
            models,
            task.x_train_scaled,
            evidential=evidential,
            device=config.device,
            batch_size=config.predict_batch_size,
        )
        disagreement = scale_robust(disagreement_raw, train_disagreement_raw)

    if strength is None:
        uncertainty = np.clip(0.50 * entropy + 0.30 * task.novelty + 0.20 * disagreement, 0.0, 1.0)
    else:
        evidence_uncertainty = np.clip(len(task.known_labels) / np.maximum(strength, 1e-6), 0.0, 1.0)
        uncertainty = np.clip(0.45 * evidence_uncertainty + 0.25 * entropy + 0.30 * task.novelty, 0.0, 1.0)
    return mean_prob, entropy, disagreement, uncertainty, all_losses


def run_unist(dataset_dir: str | Path, out_dir: str | Path, config: UnistConfig | None = None) -> dict:
    """Run UNIST on a materialized dataset directory.

    The dataset directory must contain `spatial.h5ad` and `metadata.json`. For real
    reference-supervised mapping it must also contain `sc_reference.h5ad` with a
    `cell_type` column or a metadata `sc_label_col` entry. For synthetic/gold
    tasks it can instead include `ground_truth_proportions.csv`.
    """
    config = config or UnistConfig()
    if config.model not in {"mlp_ensemble", "evidential_mlp"}:
        raise ValueError("config.model must be 'mlp_ensemble' or 'evidential_mlp'")

    dataset_dir = Path(dataset_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    task = load_task(dataset_dir, seed=config.seed)
    method = _method_name(config)
    mean_prob, entropy, disagreement, uncertainty, losses = _train_and_predict(task, config)

    confidence = mean_prob.max(axis=1)
    pred_idx = mean_prob.argmax(axis=1)
    pred_label = np.array(task.known_labels, dtype=object)[pred_idx]
    spot_ids = task.spatial.obs_names.astype(str)

    predicted = pd.DataFrame(
        {
            "spot_id": spot_ids,
            "predicted_label": pred_label,
            "confidence": confidence,
            "uncertainty": uncertainty,
            "predictive_entropy": entropy,
            "model_disagreement": disagreement,
            "novelty_score": task.novelty,
            "method": method,
        }
    )
    uncertainty_df = pd.DataFrame(
        {
            "unit_id": spot_ids,
            "unit_type": "spot",
            "uncertainty_score": uncertainty,
            "method": method,
        }
    )
    abundance = pd.concat(
        [
            pd.DataFrame(
                {
                    "spot_id": spot_ids,
                    "cell_type": label,
                    "score_or_abundance": mean_prob[:, j],
                    "method": method,
                }
            )
            for j, label in enumerate(task.known_labels)
        ],
        ignore_index=True,
    )

    metrics = {
        "method": method,
        "model": config.model,
        "n_spots": int(task.spatial.n_obs),
        "n_train_units": int(task.n_train_units),
        "training_source": task.training_source,
        "n_genes": int(task.spatial.n_vars),
        "n_classes": int(len(task.known_labels)),
        "ensemble_size": int(1 if config.model == "evidential_mlp" else config.ensemble_size),
        "epochs": int(config.epochs),
        "device": config.device,
        "final_loss_mean": float(np.mean([values[-1] for values in losses])),
        "mean_confidence": float(np.mean(confidence)),
        "mean_uncertainty": float(np.mean(uncertainty)),
        "status": "completed",
    }

    predicted.to_csv(out_dir / "predicted_labels.csv", index=False, encoding="utf-8-sig")
    uncertainty_df.to_csv(out_dir / "uncertainty.csv", index=False, encoding="utf-8-sig")
    abundance.to_csv(out_dir / "spot_by_celltype.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([metrics]).to_csv(out_dir / "metrics.csv", index=False, encoding="utf-8-sig")
    write_json(out_dir / "training_losses.json", losses)
    write_json(out_dir / "run_metadata.json", metrics)
    return metrics

