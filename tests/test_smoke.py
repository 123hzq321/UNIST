from __future__ import annotations

import json

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from unist import run_unist
from unist.api import UnistConfig
from unist.report import build_rejection_report


@pytest.mark.parametrize("model", ["evidential_mlp", "mlp_ensemble"])
def test_run_unist_reference_supervised_smoke(tmp_path, model):
    dataset = tmp_path / "dataset"
    out = tmp_path / "out"
    dataset.mkdir()

    genes = [f"gene_{i}" for i in range(12)]
    sc = ad.AnnData(
        X=np.random.default_rng(1).poisson(2, size=(60, 12)).astype("float32"),
        obs=pd.DataFrame({"cell_type": ["A"] * 30 + ["B"] * 30}, index=[f"cell_{i}" for i in range(60)]),
        var=pd.DataFrame(index=genes),
    )
    spatial = ad.AnnData(
        X=np.random.default_rng(2).poisson(2, size=(15, 12)).astype("float32"),
        obs=pd.DataFrame(index=[f"spot_{i}" for i in range(15)]),
        var=pd.DataFrame(index=genes),
    )
    sc.write_h5ad(dataset / "sc_reference.h5ad")
    spatial.write_h5ad(dataset / "spatial.h5ad")
    (dataset / "metadata.json").write_text(json.dumps({"dataset_id": "smoke"}), encoding="utf-8")

    metrics = run_unist(dataset, out, UnistConfig(model=model, epochs=2, ensemble_size=2, batch_size=16, device="cpu"))
    assert metrics["status"] == "completed"
    assert (out / "predicted_labels.csv").exists()
    assert (out / "uncertainty.csv").exists()
    predicted = pd.read_csv(out / "predicted_labels.csv")
    assert len(predicted) == 15
    assert predicted["spot_id"].is_unique
    report = build_rejection_report(out, out / "report")
    assert report["n_spots"] == 15
    policies = pd.read_csv(out / "report" / "rejection_policy.csv")
    assert ((policies.n_kept + policies.n_rejected) == 15).all()
    assert (out / "report" / "rejection_report.md").is_file()

