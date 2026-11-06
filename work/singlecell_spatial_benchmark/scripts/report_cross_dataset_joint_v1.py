"""Aggregate-only report for the frozen cross-dataset transport experiment."""
import argparse
import json
from pathlib import Path
import shutil

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from validate_cross_dataset_joint_v1 import CORE, sha, write_json

NAMES = {'entropy_only': 'Entropy', 'novelty_only': 'Novelty',
         'disagreement_only': 'Disagreement', 'fixed_fusion': 'Fixed fusion',
         'composition_maxprob': '1 - max probability', 'composition_B_margin': 'B/Plasma margin'}
STRATA = [('GSE210616', 'frozen'), ('GSE175540', 'frozen'), ('GSE175540', 'FFPE'),
          ('GSE210041', 'FFPE'), ('GSE300676', 'FFPE')]
STRATUM_NAMES = ['TNBC: 4 patients / 8 sections', 'RCC: 12 frozen sections',
                 'RCC: 12 FFPE sections', 'RCC: 2 FFPE patients', 'LUAD: 4 cases / 8 FFPE sections']


def tls_summary(data):
    rows = []
    for (preparation, score, policy), group in data.groupby(['preparation', 'score', 'policy']):
        paired = group[(group.n_tls > 0) & (group.n_non_tls > 0)]
        valid_auc = paired.dropna(subset=['tls_b_cell_fraction_auc_full', 'tls_b_cell_fraction_auc_retained'])
        rows.append({'preparation': preparation, 'score': score, 'policy': policy,
                     'n_annotated_sections': len(group), 'n_both_class_sections': len(paired),
                     'n_valid_retained_auc_pairs': len(valid_auc),
                     'mean_tls_retention': paired.tls_retention.mean(),
                     'mean_non_tls_retention': paired.non_tls_retention.mean(),
                     'mean_tls_minus_non_tls_retention': (paired.tls_retention - paired.non_tls_retention).mean(),
                     'mean_full_b_cell_tls_auc': paired.tls_b_cell_fraction_auc_full.mean(),
                     'mean_auc_change_among_valid_pairs': valid_auc.auc_change.mean(),
                     'inference': 'descriptive_sections_not_independent_patients',
                     'truth_role': 'structure_preservation_not_mapping_error'})
    return pd.DataFrame(rows)


def figure(inventory, coverage, tls, out):
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9,
                         'svg.fonttype': 'none', 'pdf.fonttype': 42,
                         'axes.spines.right': False, 'axes.spines.top': False})
    fig = plt.figure(figsize=(14, 9), layout='constrained')
    grid = fig.add_gridspec(2, 2, width_ratios=[1.12, 1])
    ax = fig.add_subplot(grid[0, 0])
    for i, (cohort, prep) in enumerate(STRATA):
        s = inventory[(inventory.cohort == cohort) & (inventory.preparation == prep)]
        offset = np.linspace(-.11, .11, len(s))
        ax.scatter(100 * s.selected_coverage, i + offset, s=22,
                   color='#176f73' if s.technical_coverage_eligible.all() else '#bd4d40', alpha=.75)
    ax.axvline(90, color='0.4', linestyle='--', linewidth=1)
    ax.set(yticks=range(5), yticklabels=STRATUM_NAMES, xlim=(82, 101), ylim=(4.5, -.5),
           xlabel='Frozen model features present (%)', title='A  Technical input coverage')
    ax.text(90.4, 4.45, '90% gate', color='0.4', fontsize=8)

    ax = fig.add_subplot(grid[0, 1])
    values = []
    for cohort, prep in STRATA:
        s = coverage[(coverage.cohort == cohort) & (coverage.preparation == prep) & (coverage.nominal_coverage == .8)]
        values.append(s.set_index('score').loc[CORE, 'mean_coverage'].to_numpy())
    values = np.asarray(values)
    ax.imshow(values, cmap='cividis', vmin=0, vmax=1, aspect='auto')
    for i in range(len(STRATA)):
        for j in range(len(CORE)):
            ax.text(j, i, f'{values[i, j]:.1%}', ha='center', va='center',
                    color='white' if values[i, j] < .45 else 'black', fontsize=10)
    ax.set(xticks=range(4), xticklabels=[NAMES[k] for k in CORE], yticks=range(5),
           yticklabels=['TNBC frozen', 'RCC frozen', 'RCC FFPE', 'RCC 2-patient FFPE', 'LUAD FFPE'],
           title='B  Actual retention at frozen nominal 80%')
    ax.tick_params(axis='x', labelrotation=15)

    ax = fig.add_subplot(grid[1, 0])
    base = tls.drop_duplicates('sample_id')
    for i, prep in enumerate(['frozen', 'FFPE']):
        s = base[(base.preparation == prep) & (base.n_tls > 0) & (base.n_non_tls > 0)]
        offset = np.linspace(-.12, .12, len(s))
        ax.scatter(i + offset, s.tls_b_cell_fraction_auc_full, s=35,
                   color='#176f73' if prep == 'frozen' else '#bd4d40')
        ax.hlines(s.tls_b_cell_fraction_auc_full.mean(), i - .25, i + .25, color='black', linewidth=2)
    ax.axhline(.5, color='0.5', linestyle='--', linewidth=1)
    ax.set(xticks=[0, 1], xticklabels=['Frozen: 7 two-class sections', 'FFPE: 10 two-class sections'],
           xlim=(-.5, 1.5), ylim=(.3, 1.02), ylabel='B/Plasma score vs author TLS label (AUROC)',
           title='C  TLS structure control before rejection')
    ax.text(.5, .32, 'Proxy biology only; no exact mapping-error labels', ha='center', fontsize=8)

    ax = fig.add_subplot(grid[1, 1])
    summary = tls_summary(tls)
    order = CORE + ['composition_maxprob', 'composition_B_margin']
    for delta, prep, color, label in [(-.15, 'frozen', '#176f73', 'Frozen, 7 sections'),
                                     (.15, 'FFPE', '#bd4d40', 'FFPE, 10 sections')]:
        s = summary[(summary.preparation == prep) & (summary.policy == 'matched_test_80')].set_index('score').loc[order]
        ax.barh(np.arange(len(order)) + delta, s.mean_tls_minus_non_tls_retention * 100,
                height=.28, color=color, label=label)
    ax.axvline(0, color='0.3', linewidth=1)
    ax.set(yticks=range(len(order)), yticklabels=[NAMES[k] for k in order],
           xlabel='TLS minus non-TLS retention (percentage points)',
           title='D  Structure loss at matched 80% spot retention')
    ax.invert_yaxis()
    ax.legend(frameon=False, fontsize=8, loc='upper left', bbox_to_anchor=(0, -.18), ncol=2)
    fig.suptitle('Frozen joint-model transfer: broader testing reveals coverage and selection failures', fontsize=14)
    fig.savefig(out / 'cross_dataset_transport.png', dpi=155)
    fig.savefig(out / 'cross_dataset_transport.svg')
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', type=Path, required=True)
    ap.add_argument('--publish', type=Path)
    args = ap.parse_args()
    root = args.run
    complete = json.loads((root / 'COMPLETE.json').read_text())
    correction = (json.loads((root / 'metadata_correction.json').read_text())
                  if (root / 'metadata_correction.json').exists() else {'predictions_unchanged': True})
    if complete['errors'] or not complete['protected_artifacts_unchanged'] or not correction['predictions_unchanged']:
        raise ValueError('Incomplete run or integrity failure')
    inventory = pd.read_csv(root / 'sample_inventory.csv')
    coverage = pd.read_csv(root / 'cohort_coverage_summary.csv')
    tls = pd.read_csv(root / 'tls_structure_metrics.csv')
    grouped = inventory.groupby(['cohort', 'cancer', 'preparation'], as_index=False).agg(
        sections=('sample_id', 'size'), spots=('n_inferred_spots', 'sum'),
        analysis_units=('analysis_unit', 'nunique'), unit_identity_source=('unit_identity_source', 'first'),
        selected_feature_fraction=('selected_coverage', 'min'),
        coverage_eligible=('technical_coverage_eligible', 'sum'),
        novelty_saturated_fraction_mean=('novelty_clipped_at_one_fraction', 'mean'),
        common_count_mass_fraction_min=('common_count_mass_fraction', 'min'))
    grouped.to_csv(root / 'cohort_inventory.csv', index=False)
    tls_agg = tls_summary(tls)
    tls_agg.to_csv(root / 'tls_structure_summary.csv', index=False)
    figure(inventory, coverage, tls, root)
    at80 = coverage[coverage.nominal_coverage.eq(.8)].pivot(index=['cohort', 'preparation'], columns='score', values='mean_coverage')[CORE]
    brief = tls_agg[(tls_agg.policy == 'matched_test_80') & tls_agg.score.isin(CORE)][[
        'preparation', 'score', 'n_both_class_sections', 'mean_tls_retention', 'mean_non_tls_retention',
        'mean_tls_minus_non_tls_retention', 'mean_auc_change_among_valid_pairs']]
    text = '''FROZEN CROSS-DATASET JOINT-MODEL AUDIT - 7 NOVEMBER 2026

Execution and scope
42/42 additional sections; four source studies; 125,430 spatial spots; no errors.
The previous audit contained 14 sections / 39,857 spots / two other sources.
Together: 56 sections, 165,287 spots and six source studies for this frozen
joint-model audit. This is not the total historical project benchmark count.
Twenty new sections pass the existing 90% feature-coverage gate; 22 are
feature-limited diagnostics. Including the previous audit: 26 eligible sections.
These are retrospective external source-study tests, not new blinded cohorts.
The same cancer categories were used in training. This is not an unseen-cancer
experiment. Cross-study patient aliases have not been exhaustively ruled out.

No retraining, recalibration, threshold selection or score-weight changes were
performed. All three checkpoint hashes and all 14 previous prediction files
were unchanged. Replay of 32 original test cells passed (max delta 4.31e-9).

New source inventory
''' + grouped.to_string(index=False, float_format=lambda x: f'{x:.4f}') + '''

Patient accounting
GSE210616: four author-named TNBC patients, two sections each, eight local
sections out of a 43-record source series. This is not the full source series.
GSE300676: four LUAD source cases, two sections each. The CRC filename prefix
does not denote colorectal cancer; cancer type comes from GEO source metadata.
GSE210041: two author-named RCC patients, one FFPE section each. Metadata were
corrected after the run; see METADATA_CORRECTION.txt. No treatment comparison.
GSE175540: patient identities remain unresolved. Twenty-four section identifiers
must not be promoted to 24 independent patients. TLS comparisons are descriptive.

Coverage transport at the ORIGINAL nominal 80% threshold
(patient/case-balanced for breast and lung; section means for RCC)
''' + at80.to_string(float_format=lambda x: f'{100*x:.2f}%') + '''

Breast and lung confidence intervals first average sections within the four
source patients/cases, then bootstrap those four units with 5,000 draws. With
only four units, these intervals are fragile and do not establish independence
between cohorts. RCC summaries have no population confidence interval.

Interpretation
The frozen nominal coverage does not transport reliably. Lower retention is not
proof of better error detection: unlabeled queries have no exact mapping-error
truth. All 22 FFPE sections have novelty clipped at one across all spots, so the
clipped novelty score cannot rank their regions. These sections also retain
only 85.60% of the model features. Gene availability, platform, source and biology
are confounded; this is stratified transfer testing, not a causal factorial
estimate of a platform effect. Near-complete observed count-mass recovery does
not restore the unmeasured features.

TLS structure preservation control
Twenty sections have explicit binary annotations; only 17 contain both TLS and
non-TLS labels (7 frozen, 10 FFPE). Two have only non-TLS labels and one only TLS.
No AUC is assigned to a one-class subset. Three other annotation files contain
T aggregates or nonbinary labels and are excluded; one section lacks annotation.
TLS is a biological structure label, NOT a mapping-error label or exact cell
proportion. B/Plasma-TLS association is a plausibility control, not accuracy truth.

At matched test-set 80% retention, the following means use the SAME two-class
sections within each preparation. This is a retrospective ranking diagnostic,
not a deployable frozen-threshold policy or an independent downstream rescue.
''' + brief.to_string(index=False, float_format=lambda x: f'{x:.4f}') + '''

The fixed fusion preferentially removes TLS in both preparation groups. An
AUROC computed after removing a biological class is not sufficient evidence
of benefit. Report class-specific retention beside every selected-subset AUC.
For FFPE, novelty is constant after clipping, so matched 80% selection uses only
label-blind barcode-hash tie breaking; its near-neutral TLS retention is not
evidence that novelty distinguishes reliable and unreliable regions.
Composition-based scores are reported as controls; they are not uncertainty
truth. No universal superiority over entropy, novelty or disagreement is shown.

Exclusions and limits
GSE283052 CRC remains excluded because its original gene-row order has not been
verified. A generic feature list with the same length is not a valid substitute.
No new CRC pathology rescue, clinical endpoint, exact fraction ground truth,
independent clinical validation or mature-mapper comparison was completed here.
The original paper and submission release have not been modified.

Next scientific gate
Before an accuracy or rescue claim: obtain verified source patient linkage and
independent annotation; define a separate spatial calibration cohort; lock
feature support and policies there; evaluate on untouched source-held-out test
cohorts with patient-level endpoints and class-specific retention. Any future
adaptation using these 42 sections must designate them development data and
must not rebrand the present results as independent validation of that adaptation.

Sources
https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE210616
https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE175540
https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE210041
https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE300676
Source metadata were read from NCBI GEO FTP SOFT files; their URLs and SHA256
values are recorded in run_provenance_public.json. No source clinical tables,
per-spot predictions, model weights or raw data are included in the public export.
'''
    (root / 'REPORT.txt').write_text(text, encoding='utf-8')
    if (root / 'RUN_LOCK.json').exists():
        lock = json.loads((root / 'RUN_LOCK.json').read_text())
        public = {k: lock[k] for k in ['started_utc', 'script_sha256', 'protocol_sha256', 'sources',
                                      'frozen_calibration', 'label_use', 'no_training_or_threshold_refit',
                                      'source_studies_held_out_from_joint_fit', 'cross_study_patient_aliases',
                                      'not_a_new_blinded_cohort', 'expected_sections']}
        public['model_integrity'] = [{'file': Path(v['path']).name, 'sha256': v['sha256']} for v in lock['model_integrity']]
        public['protected_previous_predictions'] = [{'sample_id': Path(v['path']).parent.name, 'sha256': v['sha256']}
                                                   for v in lock['protected_previous_predictions']]
        public['public_inference_script_sha256_after_metadata_correction'] = sha(Path(__file__).with_name('validate_cross_dataset_joint_v1.py'))
        public['historical_protocol_note'] = 'Original protocol retained verbatim; metadata correction supersedes the GSE210041 identity assumption.'
    else:
        public = json.loads((root / 'run_provenance_public.json').read_text())
    write_json(root / 'run_provenance_public.json', public)
    if args.publish:
        args.publish.mkdir(parents=True, exist_ok=True)
        files = ['REPORT.txt', 'PROTOCOL.txt', 'sample_inventory.csv', 'cohort_inventory.csv',
                 'section_transport_metrics.csv', 'unit_coverage_metrics.csv', 'cohort_coverage_summary.csv',
                 'tls_structure_metrics.csv', 'tls_structure_summary.csv', 'COMPLETE.json',
                 'original_prediction_replay.json', 'run_provenance_public.json',
                 'cross_dataset_transport.png', 'cross_dataset_transport.svg']
        if (root / 'metadata_correction.json').exists():
            files.append('metadata_correction.json')
        for name in files:
            shutil.copyfile(root / name, args.publish / name)
        write_json(args.publish / 'EXPORT_MANIFEST.json', {
            'files': [{'name': name, 'sha256': sha(args.publish / name)} for name in files],
            'raw_data_or_per_spot_predictions_or_weights_included': False})
    print(json.dumps({'sections': len(inventory), 'spots': int(inventory.n_inferred_spots.sum()),
                      'tls_two_class_sections': int(((tls.drop_duplicates('sample_id').n_tls > 0) &
                                                     (tls.drop_duplicates('sample_id').n_non_tls > 0)).sum()),
                      'report': str(root / 'REPORT.txt')}))


if __name__ == '__main__':
    main()
