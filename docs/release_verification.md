# Release checks: 2026-10-20

The following checks were run locally with Python 3.10 on Windows:

- Editable package installation succeeded in a dedicated environment using the existing scientific dependencies listed in `requirements-tested.txt`.
- Four pytest tests passed: two model-to-report smoke tests, a frozen-source configuration check and a download hash check.
- Six joint-training contract tests passed: deterministic donor splitting, weighting, donor-restricted mixtures, frozen query scaling, shared control mixtures and normalization.
- A synthetic joint-training run with 1,080 cells, three synthetic sources, 18 synthetic donors and two ensemble members completed on CPU. This does not add real training data or clinical validation.
- Altering held-out expression preserved training-only feature statistics in the synthetic test.
- All eight core package Python files match the archived manuscript source bytes; hashes are in `core_source_provenance.json`.

The first CPU smoke attempt exposed the archived script's CUDA-only device handling. The release adds an explicit CPU option for small checks; the original CUDA default and scientific calculations are retained.

The full public source downloads, full-scale retraining, pretrained model release and historical external benchmark matrix were not rerun for this software release. Passing these checks does not establish complete reproduction of the manuscript.
