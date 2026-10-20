from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


def _fmt(value: float) -> str:
    return "NA" if not np.isfinite(value) else f"{value:.3f}"


def _markdown_table(df: pd.DataFrame) -> list[str]:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in df.iterrows():
        vals = []
        for col in cols:
            val = row[col]
            vals.append(f"{val:.3f}" if isinstance(val, float) and np.isfinite(val) else str(val))
        lines.append("| " + " | ".join(vals) + " |")
    return lines


def build_rejection_report(result_dir: str | Path, out_dir: str | Path | None = None, dataset_name: str | None = None) -> dict:
    result_dir = Path(result_dir)
    out_dir = Path(out_dir) if out_dir is not None else result_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    pred = pd.read_csv(result_dir / "predicted_labels.csv")
    unc_col = "uncertainty" if "uncertainty" in pred.columns else "uncertainty_score"
    if unc_col not in pred.columns:
        unc = pd.read_csv(result_dir / "uncertainty.csv").rename(columns={"unit_id": "spot_id"})
        pred = pred.merge(unc[["spot_id", "uncertainty_score"]], on="spot_id", how="left")
        unc_col = "uncertainty_score"

    pred[unc_col] = pred[unc_col].astype(float)
    quantiles = pred[unc_col].quantile([0, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0]).reset_index()
    quantiles.columns = ["quantile", "uncertainty"]
    quantiles.to_csv(out_dir / "uncertainty_quantiles.csv", index=False, encoding="utf-8-sig")

    policy_rows = []
    for keep in [0.9, 0.8, 0.7, 0.6, 0.5]:
        cutoff = float(pred[unc_col].quantile(keep))
        kept = pred[pred[unc_col] <= cutoff]
        row = {
            "keep_fraction": keep,
            "uncertainty_cutoff": cutoff,
            "n_kept": int(len(kept)),
            "n_rejected": int(len(pred) - len(kept)),
            "mean_confidence_kept": float(kept["confidence"].mean()) if "confidence" in kept else float("nan"),
            "mean_uncertainty_kept": float(kept[unc_col].mean()),
            "mean_uncertainty_rejected": float(pred[pred[unc_col] > cutoff][unc_col].mean()),
        }
        policy_rows.append(row)
    policy = pd.DataFrame(policy_rows)
    policy.to_csv(out_dir / "rejection_policy.csv", index=False, encoding="utf-8-sig")

    label_summary = (
        pred.groupby("predicted_label", dropna=False)
        .agg(n_spots=("spot_id", "count"), mean_confidence=("confidence", "mean"), mean_uncertainty=(unc_col, "mean"))
        .reset_index()
        .sort_values("n_spots", ascending=False)
    )
    label_summary.to_csv(out_dir / "label_uncertainty_summary.csv", index=False, encoding="utf-8-sig")

    dataset = dataset_name or result_dir.name
    recommended = policy[policy["keep_fraction"].eq(0.8)].iloc[0]
    report = [
        f"# UNIST Rejection Report: {dataset}",
        "",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "## Summary",
        "",
        f"- Spots: {len(pred):,}",
        f"- Mean uncertainty: {_fmt(float(pred[unc_col].mean()))}",
        f"- Median uncertainty: {_fmt(float(pred[unc_col].median()))}",
        f"- Suggested default policy: keep 80% lowest-uncertainty spots, reject {int(recommended['n_rejected']):,} high-risk spots.",
        "",
        "## Rejection Policy",
        "",
    ]
    report.extend(_markdown_table(policy))
    report.extend(["", "## Label-Level Uncertainty", ""])
    report.extend(_markdown_table(label_summary.head(20)))
    report.extend(
        [
            "",
            "## Interpretation",
            "",
            "UNIST uncertainty is a guardrail. High-uncertainty spots should be inspected before using reference-derived cell-type interpretations for downstream biological conclusions.",
        ]
    )
    (out_dir / "rejection_report.md").write_text("\n".join(report), encoding="utf-8")
    return {
        "n_spots": int(len(pred)),
        "mean_uncertainty": float(pred[unc_col].mean()),
        "recommended_keep_fraction": 0.8,
        "recommended_cutoff": float(recommended["uncertainty_cutoff"]),
        "report": str(out_dir / "rejection_report.md"),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build a lightweight UNIST uncertainty/rejection report.")
    parser.add_argument("--result-dir", required=True, help="Directory containing predicted_labels.csv from unist-map.")
    parser.add_argument("--out-dir", default=None, help="Output directory. Defaults to result-dir.")
    parser.add_argument("--dataset-name", default=None)
    args = parser.parse_args(argv)
    info = build_rejection_report(args.result_dir, args.out_dir, args.dataset_name)
    print(info)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

