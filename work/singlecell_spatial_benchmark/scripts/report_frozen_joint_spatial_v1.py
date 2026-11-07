"""Summarize all frozen-joint validation outcomes, including failures."""
from pathlib import Path
import argparse
import itertools
import json
import subprocess
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from validate_frozen_joint_spatial_v1 import sha

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / 'outputs/frozen_joint_spatial_20261107'
CORE = ['entropy_only', 'novelty_only', 'disagreement_only', 'fixed_fusion']
NAMES = {'entropy_only': 'Entropy', 'novelty_only': 'Novelty',
         'disagreement_only': 'Disagreement', 'fixed_fusion': 'Fixed fusion',
         'composition_maxprob': 'Max-probability',
         'composition_endpoint_margin': 'Endpoint margin', 'single_mlp_entropy': 'Single-member entropy'}


def exact_paired_sensitivity(metrics):
    m = metrics[(metrics.cohort == 'breast_wu2021') & (metrics.policy == 'matched_test_80')]
    pivot = m.pivot(index='sample_id', columns='score', values='retained_pathology_auc')
    rows = []
    for baseline in pivot.columns:
        if baseline == 'fixed_fusion':
            continue
        delta = (pivot.fixed_fusion - pivot[baseline]).dropna().to_numpy()
        signs = np.array(list(itertools.product([-1, 1], repeat=len(delta))))
        null = (signs * delta).mean(axis=1)
        observed = delta.mean()
        p = float(np.mean(np.abs(null) >= abs(observed) - 1e-12))
        loo = [(delta.sum() - x) / (len(delta) - 1) for x in delta]
        rows.append({'baseline': baseline, 'paired_difference': observed, 'n_sections': len(delta),
                     'sections_favoring_fusion': int((delta > 0).sum()),
                     'exact_sign_flip_p_two_sided': p, 'leave_one_section_out_min': min(loo),
                     'leave_one_section_out_max': max(loo)})
    order = np.argsort([r['exact_sign_flip_p_two_sided'] for r in rows])
    running = 0.
    for rank, i in enumerate(order):
        running = max(running, (len(rows) - rank) * rows[i]['exact_sign_flip_p_two_sided'])
        rows[i]['holm_p_six_comparisons'] = min(1., running)
    result = pd.DataFrame(rows)
    result.to_csv(OUT / 'paired_inference_sensitivity.csv', index=False)
    return result


def create_figure(metrics, inventory, paired):
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9,
                         'svg.fonttype': 'none', 'axes.spines.top': False,
                         'axes.spines.right': False})
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), layout='constrained')
    fig.suptitle('Frozen joint-reference model: external histology audit', fontsize=16)
    ax = axes[0, 0]
    colors = ['#2b7a78' if v else '#d38725' for v in inventory.technical_coverage_eligible]
    ax.bar(np.arange(len(inventory)), inventory.selected_coverage, color=colors)
    ax.axhline(.9, color='#444444', linestyle='--', linewidth=1)
    ax.set_xticks(np.arange(len(inventory)), inventory.sample_id, rotation=65, ha='right')
    ax.set_ylim(0, 1.08)
    ax.set_ylabel('Fraction of 2,000 fixed features present')
    ax.set_title('A  Feature coverage: six eligible, eight diagnostic', loc='left', fontweight='bold')

    ax = axes[0, 1]
    b = metrics[(metrics.cohort == 'breast_wu2021') & (metrics.policy == 'matched_test_80')]
    p = b.pivot(index='sample_id', columns='score', values='retained_pathology_auc')[CORE]
    full = b.groupby('sample_id').full_pathology_auc.first()
    vals = np.column_stack([full.loc[p.index].to_numpy(), p.to_numpy()])
    for row in vals:
        ax.plot(range(5), row, marker='o', alpha=.55, linewidth=1)
    ax.plot(range(5), vals.mean(axis=0), color='black', linewidth=2.5, marker='s', label='Equal-section mean')
    ax.set_xticks(range(5), ['Unfiltered'] + [NAMES[c] for c in CORE], rotation=20, ha='right')
    ax.set_ylabel('Epithelial-presence pathology AUROC')
    ax.set_ylim(.35, 1)
    ax.legend(frameon=False, loc='lower left')
    ax.set_title('B  Breast: matched 80% retention, six sections', loc='left', fontweight='bold')

    ax = axes[1, 0]
    f = metrics[(metrics.policy == 'frozen_mixture_calibration_80') & metrics.score.isin(CORE)]
    coverage = f.pivot(index='sample_id', columns='score', values='coverage').reindex(inventory.sample_id)[CORE]
    im = ax.imshow(coverage, vmin=0, vmax=1, cmap='cividis', aspect='auto')
    ax.set_yticks(range(len(coverage)), coverage.index)
    ax.set_xticks(range(4), [NAMES[c] for c in CORE], rotation=20, ha='right')
    for i in range(len(coverage)):
        for j in range(4):
            v = coverage.iloc[i, j]
            ax.text(j, i, f'{100*v:.1f}', ha='center', va='center', color='white' if v < .5 else 'black', fontsize=8)
    ax.axhline(5.5, color='white', linewidth=2)
    fig.colorbar(im, ax=ax, label='Realized retention fraction', shrink=.8)
    ax.set_title('C  Frozen 80% threshold does not transfer', loc='left', fontweight='bold')

    ax = axes[1, 1]
    d = paired[(paired.cohort == 'breast_wu2021') & (paired.policy == 'matched_test_80')].copy()
    desired = ['entropy_only', 'novelty_only', 'disagreement_only', 'composition_maxprob', 'composition_endpoint_margin']
    d['baseline'] = d.comparison.str.removeprefix('fusion_minus_')
    d = d.set_index('baseline').reindex(desired)
    for i, row in enumerate(d.itertuples()):
        ax.plot([row.ci_low, row.ci_high], [i, i], color='#666666', linewidth=2)
        ax.scatter(row.paired_auc_difference, i, color='#b34d35', s=45, zorder=3)
    ax.axvline(0, color='black', linestyle='--', linewidth=1)
    ax.set_yticks(range(len(desired)), [NAMES[c] for c in desired])
    ax.invert_yaxis()
    ax.set_xlabel('AUROC difference: fusion minus comparator\n95% section-bootstrap intervals; exploratory')
    ax.set_title('D  No demonstrated fusion advantage', loc='left', fontweight='bold')
    fig.savefig(OUT / 'frozen_joint_validation.png', dpi=180)
    fig.savefig(OUT / 'frozen_joint_validation.svg')
    plt.close(fig)


def main():
    global OUT
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=Path, default=OUT)
    OUT = ap.parse_args().out.resolve()
    complete = json.loads((OUT / 'COMPLETE.json').read_text())
    lock = json.loads((OUT / 'RUN_LOCK.json').read_text())
    replay = json.loads((OUT / 'original_prediction_replay.json').read_text())
    metrics = pd.read_csv(OUT / 'section_metrics_all.csv')
    inventory = pd.read_csv(OUT / 'sample_inventory.csv')
    summary = pd.read_csv(OUT / 'section_bootstrap_summary.csv')
    paired = pd.read_csv(OUT / 'paired_section_comparisons.csv')
    count_mass = pd.read_csv(OUT / 'normalization_count_mass_diagnostic.csv')
    for artifact in lock['model_integrity']:
        assert sha(artifact['path']) == artifact['sha256'], artifact['path']
    runner = Path(__file__).with_name('validate_frozen_joint_spatial_v1.py')
    assert sha(runner) == lock['script_sha256']
    assert not complete['failed_sections']
    assert inventory.n_inferred_spots.sum() == complete['total_spots']
    assert inventory.sample_id.nunique() == 14
    for row in inventory.itertuples():
        prediction = pd.read_csv(OUT / row.sample_id / 'predictions.csv.gz')
        probability_columns = [c for c in prediction if c.startswith('proportion_')]
        assert not prediction.barcode.duplicated().any()
        assert np.isfinite(prediction[probability_columns].to_numpy()).all()
        np.testing.assert_allclose(prediction[probability_columns].sum(axis=1), 1., atol=2e-6)
        assert len(prediction) == row.n_inferred_spots
    sensitivity = exact_paired_sensitivity(metrics)
    create_figure(metrics, inventory, paired)
    breast = summary[(summary.cohort == 'breast_wu2021') & (summary.policy == 'matched_test_80')]
    breast_auc = breast[breast.metric == 'retained_pathology_auc'].set_index('score')
    unfiltered = float(breast[breast.metric == 'full_pathology_auc'].section_mean.iloc[0])
    difference = paired[(paired.cohort == 'breast_wu2021') & (paired.policy == 'matched_test_80')
                        & (paired.comparison == 'fusion_minus_entropy_only')].iloc[0]
    fusion_coverage = metrics[(metrics.cohort == 'breast_wu2021') & (metrics.score == 'fixed_fusion')
                             & (metrics.policy == 'frozen_mixture_calibration_80')].coverage.mean()
    lines = [
        'UNIST frozen joint-reference spatial validation - 7 November 2026',
        'Analysis: 20261107-v2. Results include negative outcomes. No manuscript edits or new submission.', '',
        'WHAT ACTUALLY RAN',
        f"All {complete['completed_sections']} sections completed, {complete['total_spots']:,} spots. No failed sections.",
        'Six Wu breast sections (15,611 spots) met the predeclared feature-coverage gate.',
        'Three kidney and five lung TLS sections (24,246 spots) were diagnostic only:',
        '1,711/2,000 fixed features were available (85.55%, below the 90% gate).',
        'Breast retained 1,998/2,000 fixed features (99.90%). This is technical eligibility,',
        'not proof of adequate expression-domain matching or independent patient identities.', '',
        'FROZEN MODEL AND TECHNICAL CHECKS',
        'All three archived checkpoint hashes matched before and after the run.',
        'Features, centroids, weights, score scales and rejection thresholds were not refitted.',
        f"32-cell original prediction replay: maximum probability difference {replay['max_abs_probability_difference']:.3g}.",
        'Read full matrices, resolve unambiguous training-reference gene aliases, use the',
        'original common-gene CP10K rule, and join author labels by exact barcode.',
        'The first attempt failed on plain-text files named .gz; those failures and the',
        'old runner are retained separately. The v2 rerun corrected format handling only.', '',
        'POST-RUN NORMALIZATION DIAGNOSTIC (NO PREDICTIONS CHANGED)',
        f"The fixed common-gene denominator accounts for {count_mass.common_gene_count_mass_fraction.min():.2%} to",
        f"{count_mass.common_gene_count_mass_fraction.max():.2%} of observed counts across the 14 sections.",
        'Thus gene-name resolution did not discard a large observed count mass. This does',
        'not recover unmeasured panel genes or establish biological calibration transfer.', '',
        'PRIMARY COVERAGE-ELIGIBLE BREAST ENDPOINT',
        'AUROC ranks epithelial-present versus non-epithelial author pathology.',
        'It is NOT exact cell-fraction error, cancer diagnosis accuracy or OOD AUROC.',
        f'Unfiltered equal-section mean AUROC: {unfiltered:.4f}.',
        'At retrospective matched 80% within-section retention:',
    ]
    for score in CORE + ['composition_maxprob', 'composition_endpoint_margin', 'single_mlp_entropy']:
        row = breast_auc.loc[score]
        lines.append(f"  {NAMES[score]:26s} {row.section_mean:.4f} [{row.ci_low:.4f}, {row.ci_high:.4f}], n={int(row.n_sections)} sections")
    lines += [
        f'Paired fusion minus entropy AUROC: {difference.paired_auc_difference:.4f}',
        f'  95% section-bootstrap interval [{difference.ci_low:.4f}, {difference.ci_high:.4f}].',
        'These small-sample bootstrap intervals are exploratory, not a superiority claim.',
        'Additional exact paired sign-flip tests, six-comparison Holm adjustment and',
        'leave-one-section-out sensitivity are in paired_inference_sensitivity.csv.',
        'They were added after seeing outcomes and are sensitivity analyses, not preregistered tests.',
        sensitivity.to_string(index=False), '',
        'THRESHOLD TRANSPORT AND COMPLEXITY CONTROLS',
        f'Original nominal-80% fusion threshold retained {fusion_coverage:.1%} of breast spots on an equal-section basis.',
        'Retained pathology AUROC was undefined in one breast section for several frozen',
        'threshold policies because both endpoint classes no longer remained. Undefined',
        'results were not converted to zero or silently called successful rejection.',
        'In all eight FFPE TLS sections, rescaled novelty saturated at 1 and its original',
        'threshold retained no spots. This is a transport/feature-panel failure diagnostic,',
        'not evidence that every spot is biologically unreliable.',
        'TLS and mixed pathology are complexity controls, not positive mapping-error labels.', '',
        'INTERPRETATION',
        'This adds real spatial testing of the joint-reference weights, which was missing.',
        'It does not demonstrate that fixed fusion improves pathology-consistent retention.',
        'The breast mean fell from unfiltered 0.7273 to fusion-retained 0.7085; entropy and',
        'novelty were competitive or better. No stronger performance claim is supported.',
        'A pathology proxy can disagree with estimated mixtures without proving composition',
        'error. Removing observations also changes case mix; matched random-retention',
        'controls and class counts are reported, not merely a favorable retained AUROC.', '',
        'LIMITATIONS AND WHAT REMAINS',
        '- Six coverage-eligible sections comprise ONE breast source cohort, not six cohorts.',
        '- The TLS source is a second source cohort; eight sections are not eight cohorts.',
        '- Public study provenance differs from training, but patient aliases are unresolved.',
        '- These cohorts had legacy-model exposure; this is retrospective, not a new blinded holdout.',
        '- The normalization universe is only partly present; unmeasured/filtered genes and',
        '  unresolved symbols remain a transport concern despite high model-feature coverage.',
        '- No new exact proportion truth, independent CRC rescue or mature-mapper rerun is supplied.',
        '- All ranking controls act on the same ensemble composition output; single-member',
        '  entropy is a score control here, not a separately evaluated mapping algorithm.',
        '- Unseen same-platform spatial development/calibration and genuinely untouched test',
        '  cohorts are still required before proposing revised score weights or thresholds.',
        '- Keep these 14 sections frozen as failure/transport evidence; do not tune on them',
        '  and then relabel the improved results as independent validation.', '',
        'NEXT SCIENTIFIC DECISION',
        'The current evidence supports an audit/limitations study, not a universal confidence',
        'guardrail claim. Any revised model needs a separate development cohort, compatible',
        'measurement panels and locked prospective test criteria. More similarly mismatched',
        'datasets alone do not resolve this failure.', '',
        'FILES',
        'PROTOCOL.txt and RUN_LOCK.json: design, source list, runtime, hashes, frozen thresholds.',
        'sample_inventory.csv and per-section input_audit.json: coverage, joins and provenance.',
        '*/predictions.csv.gz: compositions, component scores, labels and retention decisions.',
        'section_metrics_all.csv: every section, including technically ineligible diagnostics.',
        'coverage_eligible_statistics/: only the six coverage-eligible breast sections.',
        'section_bootstrap_summary.csv and paired_section_comparisons.csv: no spot-pooled tests.',
        'histology_associations.csv: artifact and complexity associations, explicitly not exact truth.',
        'frozen_joint_validation.png/.svg: summary with editable SVG text.', '',
        'normalization_count_mass_diagnostic.csv: observed-count denominator coverage.', '',
        'SOURCES',
        'https://zenodo.org/records/4739739',
        'https://zenodo.org/records/14620362',
        'https://github.com/navinlabcode/tnbc-chemo', '',
        'STORAGE',
        'Original data and model files are read only. Outputs are stored separately.',
    ]
    (OUT / 'REPORT.txt').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    tests = subprocess.run([sys.executable, '-B', str(Path(__file__).with_name('test_frozen_joint_spatial_v1.py'))],
                           capture_output=True, text=True, check=False)
    (OUT / 'unit_tests.txt').write_text(tests.stdout + tests.stderr, encoding='utf-8')
    if tests.returncode:
        raise RuntimeError('Validation unit tests failed')
    verification = {'model_and_calibration_hashes_rechecked': True, 'runner_matches_locked_hash': True,
                    'all_prediction_probabilities_finite_and_sum_to_one': True,
                    'section_counts_reconciled': True, 'unit_tests_passed': 9,
                    'no_claim_of_new_independent_patient_ids': True,
                    'no_new_model_training': True, 'no_manuscript_changes': True}
    (OUT / 'FINAL_CHECKS.json').write_text(json.dumps(verification, indent=2), encoding='utf-8')
    print(json.dumps(verification, indent=2))
    print('REPORT', OUT / 'REPORT.txt')


if __name__ == '__main__':
    main()
