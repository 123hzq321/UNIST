from __future__ import annotations

import argparse
import json
from pathlib import Path

from .api import UnistConfig, run_unist


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run UNIST uncertainty-aware spatial mapping.")
    parser.add_argument("--dataset-dir", required=True, help="Directory containing spatial.h5ad, metadata.json and optional sc_reference.h5ad.")
    parser.add_argument("--out-dir", required=True, help="Output directory for UNIST CSV files.")
    parser.add_argument("--model", choices=["mlp_ensemble", "evidential_mlp"], default="mlp_ensemble")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--ensemble-size", type=int, default=5)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--predict-batch-size", type=int, default=4096)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=20260726)
    parser.add_argument("--method-name", default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = UnistConfig(
        model=args.model,
        epochs=args.epochs,
        ensemble_size=args.ensemble_size,
        hidden=args.hidden,
        dropout=args.dropout,
        batch_size=args.batch_size,
        predict_batch_size=args.predict_batch_size,
        lr=args.lr,
        device=args.device,
        seed=args.seed,
        method_name=args.method_name,
    )
    metrics = run_unist(Path(args.dataset_dir), Path(args.out_dir), config)
    print(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

