# Reproducibility scope

This release has two distinct entry points:

- `unist-map` and `unist-report`: the archived UNIST 0.1.0 spatial mapping package.
- `work/singlecell_spatial_benchmark/scripts/`: the separate donor-disjoint joint-reference experiment used in the September 2026 manuscript revision.

The two paths do not use the same fitted weights. The older spatial benchmarks do not validate the joint-reference model. This repository is a source-code release, not a claim that every historical analysis has been independently rerun.

## Synthetic checks

From the repository root, in an environment with the package installed:

```bash
python -m pytest tests -q
python work/singlecell_spatial_benchmark/scripts/test_joint_training_v1.py --smoke
```

The smoke test generates synthetic cells and donor IDs, prepares features, trains a small model and checks that changing held-out expression does not change training-derived feature statistics. It is not biological validation. Temporary files use the environment's temporary directory; set `TMPDIR` on Unix or `TEMP` and `TMP` on Windows if needed.

## Locked public inputs

`configs/joint_sources_locked.json` identifies the three exact CELLxGENE asset versions, expected sizes, counts and SHA256 values used locally. These hashes are our checksums, not independently issued repository checksums. The source repositories' access and redistribution conditions still apply. Raw data, donor/cell-level tables and pretrained weights are not included in this code release.

Choose an empty data directory with adequate storage. The raw downloads total about 5.4 GB; prepared arrays and model outputs require additional space.

```bash
python scripts/fetch_locked_sources.py --root /path/to/joint_data --download
```

Without `--download`, this command only validates files already placed under `raw/`. It does not silently substitute the latest collection release. The older `fetch_joint_training_sources_v1.py` is retained for provenance but queries live metadata; do not use it to reproduce the frozen study.

## Joint-reference experiment

Set `UNIST_JOINT_ROOT` to the chosen data directory before running the study scripts. In PowerShell:

```powershell
$env:UNIST_JOINT_ROOT = 'D:\CodexData\datasets\UNIST_joint'
python work/singlecell_spatial_benchmark/scripts/run_joint_training_pipeline_v1.py
python work/singlecell_spatial_benchmark/scripts/run_compute_matched_joint_controls_v1.py
python work/singlecell_spatial_benchmark/scripts/audit_joint_evaluation_v1.py
```

On Unix, use `export UNIST_JOINT_ROOT=/path/to/joint_data`. The default is `joint_training_data` in the working directory. Preparation and training also accept `--root`; wrappers inherit the environment setting.

The original models, seeds, splits, objectives and score calculations are unchanged. Release-only changes make data paths portable and add version-locked retrieval, tests, documentation and an explicit CPU option for small verification runs. The full pipeline still defaults to CUDA. CPU and GPU runs need not be numerically identical. The exploratory source-inventory audit still refers to historical local archives and is not required for the joint training pipeline.

`requirements-tested.txt` records the environment used for release checks. Other combinations, especially newer AnnData APIs and different PyTorch/CUDA builds, have not been validated by this release. Full retraining is not part of the release smoke tests, and numerical identity across devices is not guaranteed.

## Interpretation limits

- The reported quantile-retention policies rank the supplied query. They are not prospective coverage or clinical-safety guarantees.
- High entropy can describe a real cellular mixture. A high uncertainty score is not itself a pathology label or proof of artefact.
- Score fusion is not consistently better than entropy, novelty or disagreement. Comparisons must match the endpoint and retained coverage.
- External benchmark packages, the full legacy runners, raw cohorts and fitted weights are not bundled here. Code availability alone does not make the manuscript's complete data-and-model reproduction package finished.
