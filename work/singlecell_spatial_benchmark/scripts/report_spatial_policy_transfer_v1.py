"""Report all locked policies, separating calibration, new testing and known TLS."""
import argparse
import json
from pathlib import Path
import shutil

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from validate_frozen_joint_spatial_v1 import sha, write_json

POLICIES = ['original_synthetic', 'spatial_pooled', 'spatial_class_conditional']
POLICY_NAMES = ['Original', 'Spatial pooled', 'Class conditional']
CORE = ['entropy_only', 'novelty_only', 'disagreement_only', 'fixed_fusion']
SCORES = CORE + ['novelty_raw', 'disagreement_raw', 'composition_maxprob']
SCORE_NAMES = ['Entropy', 'Clipped novelty', 'Clipped disagreement', 'Fixed fusion',
               'Raw novelty', 'Raw disagreement', '1 - max probability']


def summaries(metrics, bio, root):
    cols = ['role', 'cohort', 'preparation', 'score', 'policy']
    transport = metrics.groupby(cols, as_index=False).agg(
        sections=('sample_id', 'nunique'), mean_coverage=('hypothetical_coverage', 'mean'),
        mean_abs_coverage_error=('coverage_minus_target', lambda x: float(x.abs().mean())),
        min_coverage=('hypothetical_coverage', 'min'), max_coverage=('hypothetical_coverage', 'max'),
        eligible_sections=('input_coverage_eligible', 'sum'))
    transport.to_csv(root / 'coverage_summary.csv', index=False)
    calibration = metrics[metrics.role.eq('calibration_in_sample_not_test')]
    units = calibration.groupby(cols + ['analysis_unit'], as_index=False).hypothetical_coverage.mean()
    unit_summary = units.groupby(cols, as_index=False).agg(
        n_units=('analysis_unit', 'nunique'), patient_balanced_coverage=('hypothetical_coverage', 'mean'))
    unit_summary.to_csv(root / 'calibration_patient_summary.csv', index=False)
    bio = bio.copy()
    bio['auc_change'] = bio.b_tls_auc_retained - bio.b_tls_auc_full
    biological = bio.groupby(['preparation', 'score', 'policy'], as_index=False).agg(
        n_two_class_sections=('sample_id', 'nunique'), mean_tls_retention=('tls_retention', 'mean'),
        mean_non_tls_retention=('non_tls_retention', 'mean'), mean_retention_gap=('tls_minus_non_tls', 'mean'),
        valid_retained_auc_sections=('b_tls_auc_retained', 'count'), mean_auc_change=('auc_change', 'mean'))
    biological.to_csv(root / 'retrospective_tls_summary.csv', index=False)
    return transport, unit_summary, biological


def draw(metrics, summary, bio, root):
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9, 'svg.fonttype': 'none',
                         'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(2, 2, figsize=(13, 8.6), layout='constrained')
    test = summary[summary.role.eq('new_source_joint_model_test')]
    gastric = test[test.cohort.eq('GSE251950')]
    matrix = gastric.pivot(index='score', columns='policy', values='mean_coverage').reindex(index=SCORES, columns=POLICIES)
    ax = axes[0, 0]
    ax.imshow(matrix, vmin=0, vmax=1, cmap='cividis', aspect='auto')
    for i in range(len(SCORES)):
        for j in range(len(POLICIES)):
            v = matrix.iloc[i, j]
            ax.text(j, i, f'{v:.1%}' if np.isfinite(v) else 'Not defined', ha='center', va='center',
                    color='white' if np.isfinite(v) and v < .45 else 'black')
    ax.set(xticks=range(3), xticklabels=POLICY_NAMES, yticks=range(7), yticklabels=SCORE_NAMES,
           title='A  Gastric transfer: mean retention across 10 sections')

    ax = axes[0, 1]
    s = metrics[(metrics.cohort == 'GSE251950') & (metrics.score == 'fixed_fusion')]
    wide = s.pivot(index='sample_id', columns='policy', values='hypothetical_coverage').reindex(columns=POLICIES)
    for row in wide.to_numpy():
        ax.plot(range(3), row * 100, color='#8da6ad', linewidth=.8, marker='o', markersize=3, alpha=.75)
    ax.plot(range(3), wide.mean().to_numpy() * 100, color='#176f73', marker='o', linewidth=2.5, label='Section mean')
    ax.axhline(80, color='#9c3a32', linestyle='--', label='Calibration target')
    ax.set(xticks=range(3), xticklabels=POLICY_NAMES, ylim=(0, 103), ylabel='Retained spots (%)',
           title='B  Fixed-fusion retention varies between sections')
    ax.legend(frameon=False, fontsize=8, loc='upper left')

    ax = axes[1, 0]
    s = bio[(bio.preparation == 'frozen') & bio.score.isin(CORE)]
    m = s.pivot(index='score', columns='policy', values='mean_retention_gap').reindex(index=CORE, columns=POLICIES) * 100
    ax.imshow(m, cmap='RdBu', vmin=-50, vmax=50, aspect='auto')
    for i in range(4):
        for j in range(3):
            v = m.iloc[i, j]
            ax.text(j, i, f'{v:+.1f} pp', ha='center', va='center', color='white' if abs(v) > 30 else 'black')
    ax.set(xticks=range(3), xticklabels=POLICY_NAMES, yticks=range(4), yticklabels=SCORE_NAMES[:4],
           title='C  Known RCC TLS: retention gap, 7 frozen sections')
    ax.set_xlabel('TLS minus non-TLS; retrospective, not independent rescue')

    ax = axes[1, 1]
    s = test[(test.score == 'fixed_fusion') & test.cohort.ne('GSE251950')]
    for offset, cohort, color, label in [(-.18, 'GSE250163', '#176f73', 'RCC FFPE: 1 section'),
                                        (.18, 'GSE280318', '#b65044', 'CRC HD FFPE: 2 sections')]:
        val = s[s.cohort.eq(cohort)].set_index('policy').reindex(POLICIES).mean_coverage
        ax.bar(np.arange(3) + offset, val * 100, width=.34, color=color, label=label)
    ax.axhline(80, color='0.4', linestyle='--', linewidth=1)
    ax.set(xticks=range(3), xticklabels=POLICY_NAMES, ylim=(0, 115), ylabel='Hypothetical retention (%)',
           title='D  Feature-limited inputs: review only')
    ax.legend(frameon=False, fontsize=8, loc='upper left', bbox_to_anchor=(0, -.13))
    fig.suptitle('Spatial calibration changes selection but does not establish transferable reliability', fontsize=13)
    fig.savefig(root / 'spatial_policy_transfer.png', dpi=155)
    fig.savefig(root / 'spatial_policy_transfer.svg')
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', type=Path, required=True)
    ap.add_argument('--publish', type=Path)
    args = ap.parse_args()
    root = args.run
    complete = json.loads((root / 'COMPLETE.json').read_text())
    if complete['errors'] or not complete['policies_and_protected_artifacts_unchanged']:
        raise ValueError('Do not present a failed run as complete')
    if complete['new_sections'] != 13 or complete['new_spots'] != 1123475:
        raise ValueError('This report is scoped to the locked 13-section source snapshot')
    metrics = pd.read_csv(root / 'section_policy_metrics.csv')
    bio = pd.read_csv(root / 'retrospective_tls_metrics.csv')
    inventory = pd.read_csv(root / 'new_source_inventory.csv')
    summary, cal, biology = summaries(metrics, bio, root)
    draw(metrics, summary, biology, root)
    policies = json.loads((root / 'fitted_policies.json').read_text())
    support = []
    for prep, scores in policies.items():
        for label, values in scores['entropy_only']['conditional'].items():
            support.append({'preparation': prep, 'predicted_class': label,
                            **{k: v for k, v in values.items() if k != 'rule'}})
    pd.DataFrame(support).to_csv(root / 'conditional_calibration_support.csv', index=False)
    test = summary[summary.role.eq('new_source_joint_model_test')]
    main_table = test[test.score.eq('fixed_fusion')][['cohort', 'preparation', 'sections', 'policy', 'mean_coverage', 'mean_abs_coverage_error', 'eligible_sections']]
    tls_table = biology[(biology.preparation == 'frozen') & biology.score.eq('fixed_fusion')]
    report = '''SPATIAL COVERAGE CALIBRATION AND SOURCE-HELD-OUT POLICY TRANSFER
7 November 2026

Completed execution
Calibration: 16 existing sections, four TNBC patients and four LUAD cases.
New frozen-model test: 13 sections from three other sources, 1,123,475 spots.
- GSE251950: 10 gastric sections / 32,291 spots; all locally available sections.
- GSE250163: one RCC section / 3,388 spots; a local subset, not the entire series.
- GSE280318: two CRC HD sections / 1,087,796 nonzero-common-count 8-um bins.
  Eighty-five zero-common-count bins were excluded (22 P5 and 63 P2).
Ten gastric sections pass the pre-existing >=90% feature coverage gate.
RCC has 85.60% and CRC 89.35% of selected model features. All three are review-only.
The CRC library attributes specify Visium HD v1; its barcodes specify 8-um bins.
Calibration uses conventional FFPE Visium for that stratum. Matching preservation
does NOT make HD and conventional Visium the same assay or resolution.

The frozen mapper has now been applied to 69 sections / 1,288,762 spots from nine
source studies across these three audit rounds. This is a mapper-exposure ledger,
not 69 independent policy-validation samples. Sixteen earlier sections now serve
as calibration data. Twenty-four earlier RCC sections are reused for known TLS
diagnostics and are NOT counted as new samples. Millions of correlated bins do
not substitute for independent patients, cohorts or outcome labels.

What changed and what stayed frozen
No neural weights, selected genes, normalization, centroids, fusion coefficients
or original thresholds changed. New policies use unlabeled spatial calibration
scores only. New thresholds were locked before all 13 new-source predictions.
Policy alternatives are original synthetic-mixture thresholds, patient-balanced
spatial pooled thresholds and predicted-class-conditional thresholds. Unsupported
conditional classes fall back to pooled calibration. Raw novelty/disagreement,
entropy and composition confidence are all reported, with a hash-random control.
This is empirical coverage calibration, NOT error-risk calibration or a conformal
guarantee. Constant-score quantile ties are broken by label-blind hashing. Such
random retention does not restore risk discrimination. There are only four
calibration patients/cases per preparation, with one source per preparation.
Source, cancer category, platform, resolution and preparation are confounded.

New-source fixed-fusion results (equal section means, not spot pooled)
''' + main_table.to_string(index=False, float_format=lambda v: f'{v:.4f}') + '''

Interpretation
Gastric retention moves toward the target after calibration but remains variable
and well below 80% on average. This does not establish better mapping accuracy.
On the feature-limited new FFPE sources, recalibration allows nearly all spots
under a hypothetical threshold policy; the input gate still disables automatic
interpretation. Over-retention is also failed coverage transfer. Neither accepting
everything nor rejecting everything demonstrates a useful uncertainty estimator.
All scores and policies are in coverage_summary.csv; do not select fixed fusion
as the winner based on these results. No exact error or cell-fraction truth was
available for the new sources, so no error AUROC or biological rescue is claimed.

TLS structure preservation: explicitly retrospective
The preceding run already revealed TLS loss in GSE175540. These outcomes motivated
the new policy family. Although no TLS labels entered numerical calibration, this
is a known-outcome diagnostic, not independent confirmation. Patient IDs remain
unresolved; seven frozen and ten FFPE sections have both TLS and non-TLS labels.
Fixed-fusion descriptive results on the seven frozen two-class sections:
''' + tls_table.to_string(index=False, float_format=lambda v: f'{v:.4f}') + '''

Class conditioning reduces the observed TLS/non-TLS selection imbalance in this
retrospective subset. This is a hypothesis to test, not evidence that mapping
errors were corrected. Retained-subset AUC changes are sensitive to selection;
Policies have different realized test retention, so these are not equal-coverage
superiority comparisons. A smaller TLS gap can also reflect changed composition.
one-class retained subsets are missing. AUROC means can have different usable
section counts, which are explicitly reported. The entropy and raw-score policies
receive the same calibration treatment and must be compared before any incremental
value claim. No post-result score selection or additional tuning was performed.

Integrity and recovery
Original 32-cell prediction replay passed. All original model artifacts and the
16 calibration prediction files match their locked hashes. Fitted policies were
not changed during the test. Two large CRC inputs initially stopped at progress
logging because a path string was used as a Path object. Only this path conversion
was fixed; only those two failed inputs were retried. The initial failure records,
executed source and original RUN_LOCK are retained; resume_record.json records
the original and corrected script hashes. No numerical fitting changed.
37 tests pass, including streaming 50,001 spots through the string-path reader.

Limits and the next evidence requirement
All three new sources had legacy-model exposure; this is not a blinded study.
Patient identities and cross-source aliases have not been fully established, so
new-source means are descriptive, with no spot-level inferential statistics.
The gastric GC6/GC6-PM names alone are not verified donor linkage. No survival,
treatment effect or exact fraction truth is inferred. The unrelated 10x/Zenodo
CRC histology annotation was not assigned to GSE280318 by barcode coincidence.

A separate, source-held-out dataset with verified array-to-patient mapping and
manual histology annotations is still required for independent biological rescue.
The TNBC Zenodo resource distinguishes annotBySpot/annotsBySpot pixel annotations
from classifAll expression-derived classifications. The latter cannot replace
independent pathology truth. That independent manual-label experiment is not
completed in this report. Future model or policy changes using these new test
outcomes must reclassify the sources as development data and obtain another test.

Reproduction and data stewardship
Only aggregate reports, policy parameters, source hashes, code and figures are
public. No new per-spot prediction files, raw matrices, clinical tables, patient
records, mapper checkpoints or manuscript drafts were uploaded. D: remains read
only because of filesystem warnings; aggregate recovery outputs are separate.

Sources
https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE251950
https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE250163
https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE280318
https://zenodo.org/records/8135722
Source GEO protocols were retrieved from the authoritative FTP SOFT endpoint;
URLs and checksums are preserved in run_provenance_public.json.
'''
    (root / 'REPORT.txt').write_text(report, encoding='utf-8')
    if (root / 'RUN_LOCK.json').exists():
        lock = json.loads((root / 'RUN_LOCK.json').read_text())
        public = {k: v for k, v in lock.items() if k not in ['model_integrity', 'calibration_prediction_hashes', 'test_samples']}
        public['model_integrity'] = [{'file': Path(v['path']).name, 'sha256': v['sha256']} for v in lock['model_integrity']]
        public['calibration_prediction_hashes'] = [{'sample_id': Path(v['path']).parent.name, 'sha256': v['sha256']}
                                                 for v in lock['calibration_prediction_hashes']]
        public['test_samples'] = [{k: v for k, v in row.items() if k != 'matrix'} for row in lock['test_samples']]
        write_json(root / 'run_provenance_public.json', public)
    if args.publish:
        files = ['PROTOCOL.txt', 'REPORT.txt', 'COMPLETE.json', 'fitted_policies.json',
                 'new_source_inventory.csv', 'section_policy_metrics.csv', 'retrospective_tls_metrics.csv',
                 'coverage_summary.csv', 'calibration_patient_summary.csv', 'retrospective_tls_summary.csv',
                 'conditional_calibration_support.csv', 'run_provenance_public.json',
                 'original_prediction_replay.json', 'spatial_policy_transfer.png', 'spatial_policy_transfer.svg']
        if (root / 'resume_record.json').exists():
            files.append('resume_record.json')
        args.publish.mkdir(parents=True, exist_ok=True)
        for name in files:
            shutil.copyfile(root / name, args.publish / name)
        write_json(args.publish / 'EXPORT_MANIFEST.json', {
            'files': [{'name': name, 'sha256': sha(args.publish / name)} for name in files],
            'no_raw_data_or_per_spot_predictions_or_mapper_weights': True})
    print(json.dumps({'new_sections': len(inventory), 'new_spots': int(inventory.n_spots.sum()),
                      'report': str(root / 'REPORT.txt')}))


if __name__ == '__main__':
    main()
