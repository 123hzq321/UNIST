from __future__ import annotations

import argparse
import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd


def main() -> int:
    parser = argparse.ArgumentParser(description="Create a tiny reference-supervised UNIST example dataset.")
    parser.add_argument("--out-dir", default="examples/toy_dataset")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    genes = [f"gene_{i:03d}" for i in range(40)]
    cell_types = ["Tumor", "T cell", "CAF"]
    profiles = np.vstack(
        [
            np.r_[np.full(10, 6), np.full(30, 1)],
            np.r_[np.full(10, 1), np.full(10, 6), np.full(20, 1)],
            np.r_[np.full(20, 1), np.full(10, 6), np.full(10, 1)],
        ]
    )

    sc_rows = []
    sc_labels = []
    for idx, label in enumerate(cell_types):
        values = rng.poisson(profiles[idx], size=(80, len(genes))).astype("float32")
        sc_rows.append(values)
        sc_labels.extend([label] * values.shape[0])
    sc_x = np.vstack(sc_rows)
    sc = ad.AnnData(
        X=sc_x,
        obs=pd.DataFrame({"cell_type": sc_labels}, index=[f"cell_{i}" for i in range(sc_x.shape[0])]),
        var=pd.DataFrame(index=genes),
    )

    spot_profiles = []
    for _ in range(90):
        weights = rng.dirichlet([1.5, 1.5, 1.5])
        lam = weights @ profiles
        spot_profiles.append(rng.poisson(lam).astype("float32"))
    spatial = ad.AnnData(
        X=np.vstack(spot_profiles),
        obs=pd.DataFrame(index=[f"spot_{i}" for i in range(90)]),
        var=pd.DataFrame(index=genes),
    )

    sc.write_h5ad(out / "sc_reference.h5ad")
    spatial.write_h5ad(out / "spatial.h5ad")
    (out / "metadata.json").write_text(json.dumps({"dataset_id": "toy_dataset"}, indent=2), encoding="utf-8")
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

