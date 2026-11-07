# UNIST

UNIST is a research package for auditing uncertainty in single-cell reference mapping to spatial transcriptomics.

This repository contains the UNIST 0.1.0 package and a separate donor-disjoint joint-reference experiment. See [reproducibility scope](docs/reproducibility.md) before interpreting the results. Uncertainty is endpoint-dependent, and fixed fusion is not consistently superior to individual scores.

## Frozen Joint-Model Spatial Audit

The [7 November 2026 validation study](studies/frozen_joint_spatial_20261107/README.md)
reports real spatial testing of the frozen joint-reference model: 14 sections,
39,857 spots, two source cohorts. Six breast sections passed the model-feature
coverage gate; eight kidney/lung FFPE sections are transport diagnostics only.
Fixed fusion did not improve the breast pathology endpoint over the simple
scores, and the original rejection thresholds did not preserve nominal coverage.
Negative outcomes, section-level statistics and reproduction scripts are included.
These are retrospective histology-proxy analyses, not exact cell-fraction truth
or clinical validation. Raw data and fitted weights are not included.

The [next-cohort qualification audit](studies/spatial_calibration_candidate_audit_20261107/REPORT.txt)
checked 24 additional RCC sections and recorded unresolved patient identities and
annotation schemas. It was a pre-inference inspection, not a separate model test.

The [cross-dataset expansion](studies/cross_dataset_joint_20261107/REPORT.txt)
has now tested 42 additional sections / 125,430 spots from four source studies,
including those 24 RCC sections. The combined frozen-model audit covers 56
sections / 165,287 spots from six sources, not 56 independent patient cohorts.
Twenty of the new sections pass the feature-coverage gate; 22 FFPE sections are
feature-limited diagnostics. Frozen thresholds lose nominal coverage, and fixed
fusion can preferentially reject author-annotated TLS. These negative findings
do not establish mapping accuracy or a successful biological rescue. No model
weights or thresholds were adapted to these data. See the metadata correction
record for verified identities and preparation of the two GSE210041 samples.

The [spatial policy-transfer experiment](studies/spatial_policy_transfer_20261107/README.md)
uses 16 of those sections for unlabeled, patient-balanced spatial calibration and
tests the locked policies on 13 further gastric, RCC and CRC HD sections. This adds
1,123,475 spots/bins, mostly from two HD sections, not new independent patients.
Coverage transfer remains unreliable. Known RCC TLS diagnostics suggest class
conditioning changes structure selection, but independent biological rescue has
not been demonstrated. The mapper-exposure total is now 69 sections / 1,288,762
spots from nine sources; calibration and reused diagnostic sections must not be
counted as independent policy tests. Neural weights remain unchanged.

The package is designed for a specific question: when a single-cell reference is mapped onto a spatial query, can the model flag regions or cohorts where the mapping should not be over-interpreted?

UNIST is not intended to replace mature deconvolution tools such as RCTD, cell2location, Tangram or stereoscope. It provides a complementary uncertainty layer for cross-patient, cross-platform and cross-cancer spatial tumour microenvironment interpretation.

## Installation

```bash
pip install -e .
```

For GPU support, install a PyTorch build that matches your CUDA environment before installing UNIST.

For tests and the joint-reference scripts, use `pip install -e ".[dev,study]"`.
The release-check environment is listed in `requirements-tested.txt`.

## Input Format

UNIST expects one materialized dataset directory:

```text
my_dataset/
  spatial.h5ad
  metadata.json
  sc_reference.h5ad              # for real reference-supervised mapping
```

`spatial.h5ad` contains query spots or bins with genes in `var_names`.

`sc_reference.h5ad` contains single-cell reference expression with a `cell_type` column in `obs`. If the label column has a different name, set it in `metadata.json`:

```json
{
  "dataset_id": "example_dataset",
  "sc_label_col": "celltype"
}
```

For synthetic or gold-standard pseudo-spot tasks, the dataset can instead include:

```text
ground_truth_proportions.csv
```

with columns `spot_id`, `cell_type`, and `true_proportion`.

## Quick Start

Create and run a tiny end-to-end toy example:

```bash
python examples/make_toy_dataset.py --out-dir examples/toy_dataset
unist-map --dataset-dir examples/toy_dataset --out-dir examples/toy_results/evidential --model evidential_mlp --epochs 2
unist-report --result-dir examples/toy_results/evidential --out-dir examples/toy_results/evidential/report
```

Run the ensemble model:

```bash
unist-map \
  --dataset-dir path/to/my_dataset \
  --out-dir results/my_dataset/unist_mlp_ensemble \
  --model mlp_ensemble \
  --epochs 80 \
  --ensemble-size 5 \
  --device cuda
```

Run the evidential model:

```bash
unist-map \
  --dataset-dir path/to/my_dataset \
  --out-dir results/my_dataset/unist_evidential_mlp \
  --model evidential_mlp \
  --epochs 70 \
  --device cuda
```

## Outputs

UNIST writes:

- `predicted_labels.csv`: predicted dominant cell type, confidence and uncertainty components per spot.
- `uncertainty.csv`: compact per-spot uncertainty scores.
- `spot_by_celltype.csv`: per-cell-type scores or abundance-like probabilities.
- `metrics.csv`: run-level summary.
- `run_metadata.json` and `training_losses.json`.

Build a lightweight rejection report from any UNIST result directory:

```bash
unist-report \
  --result-dir results/my_dataset/unist_evidential_mlp \
  --out-dir results/my_dataset/unist_evidential_mlp/report
```

This writes `rejection_report.md`, `rejection_policy.csv`,
`uncertainty_quantiles.csv`, and `label_uncertainty_summary.csv`.

## Minimal Python API

```python
from unist import run_unist
from unist.api import UnistConfig

metrics = run_unist(
    "path/to/my_dataset",
    "results/my_dataset/unist_mlp_ensemble",
    UnistConfig(model="mlp_ensemble", epochs=80, ensemble_size=5, device="cuda"),
)
print(metrics)
```

## Repository Layout

```text
src/unist/        installable package
tests/            smoke tests
work/             archived joint-reference study scripts
scripts/          version-locked source retrieval
configs/          source versions and label harmonization
docs/             input contract and reproducibility limitations
```

## Citation

The manuscript is in preparation. Cite the repository and the fixed release tag
`submission-2026-10-20`. No journal acceptance or archival DOI is claimed.

Repository: https://github.com/123hzq321/UNIST

The public release does not include raw patient data, manuscript drafts, account
credentials or fitted weights. The synthetic example is generated locally.

## Status

This is an alpha research package. The core API and output schema may change before v1.0.
