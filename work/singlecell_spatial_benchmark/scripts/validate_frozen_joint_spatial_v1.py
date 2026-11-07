"""Read-only external spatial audit of the frozen joint-reference ensemble.

Histology endpoints are proxies, never exact composition or mapping-error truth.
Outputs are deliberately separate from all legacy model results.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import importlib.metadata
import json
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.io import mmread
from sklearn.metrics import roc_auc_score
import torch

from train_joint_reference_v1 import MLPBackbone, predict, score_components

PROJECT = Path(__file__).resolve().parents[3]
DEFAULT_ROOT = Path(os.environ.get('UNIST_DATA_ROOT', 'external_data'))
CORE = ['entropy_only', 'novelty_only', 'disagreement_only', 'fixed_fusion']
SEED = 20261107


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def count_stream(path):
    with Path(path).open('rb') as f:
        compressed = f.read(2) == b'\x1f\x8b'
    return gzip.open(path, 'rb') if compressed else Path(path).open('rb')


def read_counts(folder):
    with count_stream(folder / 'features.tsv.gz') as f:
        features = pd.read_csv(f, sep='\t', header=None, dtype=str, keep_default_na=False)
    with count_stream(folder / 'barcodes.tsv.gz') as f:
        barcodes = pd.read_csv(f, sep='\t', header=None, dtype=str)[0]
    with count_stream(folder / 'matrix.mtx.gz') as f:
        x = mmread(f).T.tocsr().astype(np.float32)
    if x.shape != (len(barcodes), len(features)):
        raise ValueError('Matrix, feature and barcode dimensions differ')
    if barcodes.duplicated().any():
        raise ValueError('Duplicate input barcodes')
    if not np.isfinite(x.data).all() or np.any(x.data < 0):
        raise ValueError('Counts must be finite and nonnegative')
    if not np.allclose(x.data, np.round(x.data), atol=1e-5):
        raise ValueError('Raw count input is not integer-valued')
    return x, features, pd.Index(barcodes)


def align_counts(x, features, selected, common, aliases=None):
    """Keep the TRAIN-defined denominator, not the 2,000-feature sum."""
    if aliases is None:
        ids = features[0].str.split('.').str[0].to_numpy()
    else:
        ids = np.array([g.split('.')[0] if g.startswith('ENSG') else aliases.get(g, 'UNMAPPED:' + g)
                        for g in features[0]])
    gene_expression = (features[2].eq('Gene Expression').to_numpy()
                       if 2 in features else np.ones(len(ids), dtype=bool))
    common_mask = np.isin(ids, common) & gene_expression
    denom = np.asarray(x[:, common_mask].sum(axis=1)).ravel().astype(np.float64)
    selected_index = {gene: i for i, gene in enumerate(selected)}
    pairs = [(i, selected_index[g]) for i, g in enumerate(ids)
             if gene_expression[i] and g in selected_index]
    rows, cols = zip(*pairs) if pairs else ([], [])
    projection = sparse.csr_matrix((np.ones(len(pairs), dtype=np.float32), (rows, cols)),
                                   shape=(len(ids), len(selected)))
    mapped = (x @ projection).toarray().astype(np.float32)
    mapped *= (10000.0 / np.maximum(denom, 1))[:, None].astype(np.float32)
    available = set(ids[gene_expression])
    n_selected = len(set(selected) & available)
    audit = {
        'n_input_genes': len(ids), 'n_selected_present': n_selected,
        'n_selected_total': len(selected), 'selected_coverage': n_selected / len(selected),
        'n_common_present': len(set(common) & available), 'n_common_total': len(common),
        'duplicate_gene_ids': int(len(ids[gene_expression]) - len(available)),
        'missing_selected_genes': sorted(set(selected) - available),
        'zero_common_count_spots': int((denom == 0).sum()),
        'unresolved_gene_symbols': sum(str(g).startswith('UNMAPPED:') for g in ids),
    }
    return mapped, denom, audit


def join_metadata(x, barcodes, metadata):
    if not metadata.index.is_unique:
        raise ValueError('Duplicate metadata barcodes')
    present = barcodes.isin(metadata.index)
    if not present.any():
        raise ValueError('No exact barcode intersection with annotations')
    return x[present], barcodes[present], metadata.loc[barcodes[present]].copy(), int((~present).sum())


def blind_keep(values, ids, fraction=0.8):
    # Barcode hashes break clipping ties without consulting pathology labels.
    tie = np.array([int(hashlib.sha256((str(SEED) + str(v)).encode()).hexdigest()[:16], 16)
                    for v in ids], dtype=np.uint64)
    order = np.lexsort((tie, np.asarray(values)))
    keep = np.zeros(len(values), dtype=bool)
    keep[order[:int(np.floor(fraction * len(values)))]] = True
    return keep


def auc(y, values):
    y, values = np.asarray(y, dtype=float), np.asarray(values, dtype=float)
    valid = np.isfinite(y) & np.isfinite(values)
    return float(roc_auc_score(y[valid], values[valid])) if len(np.unique(y[valid])) == 2 else np.nan


def pathology_endpoints(obs, cohort):
    if cohort == 'breast_wu2021':
        label = obs['Classification'].fillna('').astype(str)
        excluded = label.isin(['Artefact', 'Necrosis', 'Uncertain', ''])
        epi = label.str.contains('Invasive cancer|Cancer trapped|DCIS|Normal glands|Normal duct|Normal \\+', regex=True)
        # All other named classes here are non-epithelial pathology categories.
        y = np.where(excluded, np.nan, epi.astype(float))
        controls = {
            'histology_artefact_proxy': label.eq('Artefact').to_numpy(dtype=float),
            'mixed_pathology_not_error': np.where(excluded, np.nan, label.str.contains('+', regex=False).astype(float)),
        }
        return y, 'epithelial_presence_pathology_AUROC', controls
    label = obs['manual_anno'].fillna('UNASSIGNED').astype(str).str.upper()
    tls_label = obs['manual_anno_tls'].fillna('UNASSIGNED').astype(str).str.upper()
    tls = label.eq('TLS') | tls_label.isin(['TLS', 'ETLS', 'MTLS'])
    immune = tls | label.isin(['INFL', 'IMM', 'IMMUNE'])
    eligible = immune | label.isin(['TUM', 'TUMOR'])
    y = np.where(eligible, immune.astype(float), np.nan)
    assigned = ~label.isin(['UNASSIGNED', 'NAN', '']) | tls
    controls = {'TLS_complexity_not_error': np.where(assigned, tls.astype(float), np.nan)}
    return y, 'immune_vs_tumor_pathology_AUROC', controls


def discover(root):
    samples = []
    breast = root / 'breast_pathology_zenodo4739739'
    for f in sorted((breast / 'metadata').glob('*_metadata.csv')):
        sid = f.stem.removesuffix('_metadata')
        samples.append({'cohort': 'breast_wu2021', 'cancer': 'breast', 'sample_id': sid,
                        'counts': breast / 'filtered_count_matrices' / (sid + '_filtered_count_matrix'),
                        'annotations': f, 'source_url': 'https://zenodo.org/records/4739739',
                        'annotation_kind': 'author_H&E_spot_pathology'})
    tls = root / 'pathology_gold_zenodo14620362_tls_visium_usz/extracted/TLS_VISIUM_USZ'
    for f in sorted((tls / 'h5ad_preprocessed').glob('*.h5ad')):
        sid = f.stem
        samples.append({'cohort': 'tls_usz2025', 'cancer': 'kidney' if sid.startswith('KC') else 'lung',
                        'sample_id': sid, 'counts': tls / '10x_Visium' / sid / 'filtered_feature_bc_matrix',
                        'annotations': f, 'source_url': 'https://zenodo.org/records/14620362',
                        'annotation_kind': 'author_manual_H&E_regions_not_aestetik_predictions'})
    return samples


def load_frozen(root, device):
    joint = root / 'joint_training_v1'
    run = joint / 'runs/joint_mlp_ensemble_v1'
    prep = joint / 'prepared'
    ready = json.loads((prep / 'READY.json').read_text())
    selected = json.loads((prep / 'selected_genes.json').read_text())
    common = json.loads((prep / 'common_genes.json').read_text())
    calibration = json.loads((run / 'synthetic_within_donor_mixtures_frozen_calibration.json').read_text())
    expected = pd.read_csv(run / 'checkpoint_integrity.csv')
    models, integrity = [], []
    for member in range(3):
        path = run / f'ensemble_member_{member}.pt'
        actual = sha(path)
        suffix = '/joint_mlp_ensemble_v1/' + path.name
        match = expected[expected.path.str.replace('\\', '/', regex=False).str.endswith(suffix)]
        if len(match) != 1 or actual != match.iloc[0].sha256:
            raise ValueError(f'Frozen checkpoint integrity failed: {path}')
        model = MLPBackbone(len(selected), len(ready['labels'])).to(device)
        with torch.serialization.safe_globals([torch.torch_version.TorchVersion]):
            saved = torch.load(path, map_location=device, weights_only=True)
        model.load_state_dict(saved['state_dict'])
        model.eval()
        models.append(model)
        integrity.append({'path': str(path), 'sha256': actual, 'matches_20260926_manifest': True})
    centroids = torch.as_tensor(np.load(run / 'train_class_centroids_log1p.npy'), device=device)
    for path in [prep / 'selected_genes.json', prep / 'common_genes.json', prep / 'READY.json',
                 run / 'train_class_centroids_log1p.npy', run / 'synthetic_within_donor_mixtures_frozen_calibration.json']:
        integrity.append({'path': str(path), 'sha256': sha(path)})
    return models, centroids, selected, common, ready, calibration, integrity


def training_gene_aliases(root):
    a = ad.read_h5ad(root / 'joint_training_v1/raw/breast_tnbc.h5ad', backed='r')
    var = a.var.copy()
    a.file.close()
    aliases = {}
    for column in ['gene_symbols', 'feature_name']:
        for gene, symbol in zip(var.index.astype(str), var[column].astype(str)):
            aliases.setdefault(symbol, set()).add(gene.split('.')[0])
    unique = {symbol: next(iter(ids)) for symbol, ids in aliases.items() if len(ids) == 1}
    ambiguous = sorted(symbol for symbol, ids in aliases.items() if len(ids) > 1)
    return unique, ambiguous


def replay_original(root, models, centroids, device):
    joint = root / 'joint_training_v1'
    cells = pd.read_csv(joint / 'prepared/cells.csv.gz', usecols=['row_index', 'eligible', 'split'])
    ids = cells.loc[cells.eligible & cells.split.eq('test'), 'row_index'].to_numpy()[:32]
    matrix = np.load(joint / 'prepared/expression_cp10k.npy', mmap_mode='r')
    x = torch.as_tensor(np.array(matrix[ids], dtype=np.float32), device=device)
    actual = predict(models, x, centroids)[0]
    with np.load(joint / 'runs/joint_mlp_ensemble_v1/author_cell_labels_test_probabilities.npz') as z:
        expected = z['probabilities'][:len(ids)]
    delta = float(np.max(np.abs(actual - expected)))
    if delta > 1e-5:
        raise ValueError(f'Original held-out prediction replay failed: {delta}')
    return {'n_cells': len(ids), 'max_abs_probability_difference': delta, 'tolerance': 1e-5,
            'passed': True, 'purpose': 'technical replay, not new validation'}


def evaluate_sample(sample, models, centroids, selected, common, ready, cal, aliases, out, device):
    sid = sample['sample_id']
    folder = out / sid
    folder.mkdir()
    if sample['cohort'] == 'breast_wu2021':
        obs = pd.read_csv(sample['annotations'], index_col=0)
    else:
        backed = ad.read_h5ad(sample['annotations'], backed='r')
        obs = backed.obs.copy()
        backed.file.close()
    x, features, barcodes = read_counts(sample['counts'])
    n_input = x.shape[0]
    x, barcodes, obs, no_annotation = join_metadata(x, barcodes, obs)
    mapped, counts, feature_audit = align_counts(x, features, selected, common, aliases)
    positive = counts > 0
    mapped, obs, barcodes = mapped[positive], obs.iloc[np.where(positive)[0]], barcodes[positive]
    p, d, n, single = predict(models, torch.as_tensor(mapped, device=device), centroids)
    scores, _ = score_components(p, d, n, cal['scales'])
    scores['single_mlp_entropy'] = -(single * np.log(np.maximum(single, 1e-12))).sum(axis=1) / np.log(p.shape[1])
    scores['composition_maxprob'] = 1 - p.max(axis=1)
    labels = ready['labels']
    epi = p[:, labels.index('Epithelial_or_author_tumor')]
    immune = p[:, [labels.index(c) for c in ['B_Plasma', 'T_NK_ILC', 'Myeloid']]].sum(axis=1)
    y, endpoint, controls = pathology_endpoints(obs, sample['cohort'])
    prediction = epi if sample['cohort'] == 'breast_wu2021' else immune
    scores['composition_endpoint_margin'] = 1 - np.abs(2 * prediction - 1)
    tech_pass = feature_audit['selected_coverage'] >= 0.9
    common_row = {k: sample[k] for k in ['cohort', 'cancer', 'sample_id']}
    common_row['technical_coverage_eligible'] = tech_pass
    report = {**common_row, **feature_audit, 'n_input_spots': n_input,
              'n_no_annotation': no_annotation, 'n_inferred_spots': len(p),
              'n_annotated_endpoint': int(np.isfinite(y).sum()),
              'label_counts': {str(k): int(v) for k, v in (obs['Classification'] if sample['cohort'] == 'breast_wu2021'
                                                        else obs['manual_anno']).value_counts().items()},
              'input_files': [{'path': str(path), 'sha256': sha(path)} for path in
                              [sample['counts'] / 'features.tsv.gz', sample['counts'] / 'barcodes.tsv.gz',
                               sample['counts'] / 'matrix.mtx.gz', sample['annotations']]],
              'novelty_clipped_at_one_fraction': float(np.mean(scores['novelty_only'] >= 1)),
              'disagreement_clipped_at_one_fraction': float(np.mean(scores['disagreement_only'] >= 1))}
    frame = pd.DataFrame({'barcode': barcodes, 'cohort': sample['cohort'], 'cancer': sample['cancer'],
                          'sample_id': sid, 'endpoint_label': y, 'endpoint_prediction': prediction,
                          'novelty_raw': n, 'disagreement_raw': d, **scores})
    for i, label in enumerate(labels):
        frame['proportion_' + label] = p[:, i]
    for key, values in controls.items():
        frame[key] = values
    if 'Classification' in obs:
        frame['pathology_annotation'] = obs.Classification.to_numpy()
    else:
        frame['pathology_annotation'] = obs.manual_anno.astype(str).to_numpy()
    rows = []
    full_auc = auc(y, prediction)
    for score, values in scores.items():
        policies = {'matched_test_80': blind_keep(values, barcodes)}
        if score in cal['thresholds']:
            policies['frozen_mixture_calibration_80'] = values <= cal['thresholds'][score]['0.8']
        for policy, keep in policies.items():
            frame[f'keep_{score}_{policy}'] = keep
            eligible = np.isfinite(y)
            kept = eligible & keep
            retained_auc = auc(y[kept], prediction[kept])
            rng = np.random.default_rng(SEED)
            available = np.flatnonzero(eligible)
            random_auc = [auc(y[ii], prediction[ii]) for ii in
                          [rng.choice(available, int(kept.sum()), replace=False) for _ in range(100)]]
            valid_random = np.asarray(random_auc)[np.isfinite(random_auc)]
            random_mean = float(valid_random.mean()) if len(valid_random) else np.nan
            row = {**common_row, 'endpoint': endpoint, 'score': score, 'policy': policy,
                   'n_spots': len(p), 'n_retained': int(keep.sum()), 'coverage': float(keep.mean()),
                   'n_endpoint': int(eligible.sum()), 'n_endpoint_retained': int(kept.sum()),
                   'n_positive_retained': int(np.sum(y[kept] == 1)), 'n_negative_retained': int(np.sum(y[kept] == 0)),
                   'full_pathology_auc': full_auc, 'retained_pathology_auc': retained_auc,
                   'auc_change': retained_auc - full_auc, 'matched_random_auc': random_mean,
                   'auc_gain_over_random': retained_auc - random_mean,
                   'unique_score_values': len(np.unique(values)),
                   'frozen_threshold': cal['thresholds'][score]['0.8'] if policy.startswith('frozen') else np.nan}
            for name, control in controls.items():
                pos = np.asarray(control) == 1
                row[name + '_retention'] = float(keep[pos].mean()) if pos.any() else np.nan
            rows.append(row)
    detection = [{**common_row, 'endpoint': endpoint_name, 'score': score,
                  'score_association_auroc': auc(values, scores[score]),
                  'n_annotated': int(np.isfinite(values).sum()), 'n_positive': int(np.nansum(values)),
                  'interpretation': 'histology association; not exact mapping-error or OOD truth'}
                 for endpoint_name, values in controls.items() for score in scores]
    frame.to_csv(folder / 'predictions.csv.gz', index=False)
    write_json(folder / 'input_audit.json', report)
    return report, rows, detection


def bootstrap_summary(metrics, out):
    summaries, differences = [], []
    fields = ['coverage', 'full_pathology_auc', 'retained_pathology_auc', 'auc_change', 'auc_gain_over_random']
    for keys, group in metrics.groupby(['cohort', 'cancer', 'endpoint', 'policy', 'score']):
        for field in fields:
            values = group[field].dropna().to_numpy()
            if not len(values):
                continue
            rng = np.random.default_rng(SEED)
            draws = rng.choice(values, (5000, len(values)), replace=True).mean(axis=1)
            summaries.append(dict(zip(['cohort', 'cancer', 'endpoint', 'policy', 'score'], keys),
                                  metric=field, section_mean=float(values.mean()),
                                  ci_low=float(np.quantile(draws, .025)) if len(values) > 1 else np.nan,
                                  ci_high=float(np.quantile(draws, .975)) if len(values) > 1 else np.nan,
                                  n_sections=len(values), statistical_unit='section, not spot or verified patient'))
    for keys, group in metrics.groupby(['cohort', 'cancer', 'endpoint', 'policy']):
        pivot = group.pivot(index='sample_id', columns='score', values='retained_pathology_auc')
        for baseline in [c for c in pivot if c != 'fixed_fusion']:
            vals = (pivot.fixed_fusion - pivot[baseline]).dropna().to_numpy()
            if not len(vals):
                continue
            rng = np.random.default_rng(SEED)
            draws = rng.choice(vals, (5000, len(vals)), replace=True).mean(axis=1)
            differences.append(dict(zip(['cohort', 'cancer', 'endpoint', 'policy'], keys),
                                    comparison='fusion_minus_' + baseline,
                                    paired_auc_difference=float(vals.mean()),
                                    ci_low=float(np.quantile(draws, .025)) if len(vals) > 1 else np.nan,
                                    ci_high=float(np.quantile(draws, .975)) if len(vals) > 1 else np.nan,
                                    n_paired_sections=len(vals),
                                    interpretation='positive favors fusion; exploratory, no multiplicity-adjusted superiority claim'))
    pd.DataFrame(summaries).to_csv(out / 'section_bootstrap_summary.csv', index=False)
    pd.DataFrame(differences).to_csv(out / 'paired_section_comparisons.csv', index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-root', type=Path, default=DEFAULT_ROOT)
    ap.add_argument('--out', type=Path, default=PROJECT / 'outputs/frozen_joint_spatial_20261107')
    ap.add_argument('--protocol', type=Path, default=PROJECT / 'studies/frozen_joint_spatial_20261107/PROTOCOL.txt')
    ap.add_argument('--device', default='cuda')
    args = ap.parse_args()
    out = args.out.resolve()
    data_root = args.data_root.resolve()
    if out == data_root or data_root in out.parents:
        raise ValueError('Keep outputs outside the read-only input data directory')
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'RUN_LOCK.json').exists():
        raise FileExistsError('Output already locked; use a new directory, never overwrite a validation run')
    if args.protocol.resolve() != (out / 'PROTOCOL.txt').resolve():
        shutil.copyfile(args.protocol, out / 'PROTOCOL.txt')
    torch.set_num_threads(4)
    models, centroids, selected, common, ready, cal, integrity = load_frozen(args.data_root, args.device)
    aliases, ambiguous = training_gene_aliases(args.data_root)
    write_json(out / 'frozen_training_gene_aliases.json', {'unique': aliases, 'ambiguous_excluded': ambiguous})
    samples = discover(args.data_root)
    if len(samples) != 14:
        raise ValueError(f'Expected six breast and eight TLS sections, found {len(samples)}')
    lock = {'started_utc': datetime.now(timezone.utc).isoformat(), 'script_sha256': sha(__file__),
            'model_integrity': integrity, 'calibration': cal, 'seed': SEED,
            'gene_aliases_sha256': sha(out / 'frozen_training_gene_aliases.json'),
            'protocol': str(out / 'PROTOCOL.txt'), 'protocol_sha256': sha(out / 'PROTOCOL.txt'),
            'feature_gate': 'at least 90% of 2000 fixed genes; others diagnostic only',
            'source_independence': 'distinct source studies; donor aliases across studies not definitively resolved',
            'historical_exposure': 'these cohorts were used with legacy models; not a prospective untouched holdout',
            'statistical_unit': 'section; no claim that every section is an independent patient',
            'no_training_or_threshold_refit': True, 'input_data_written': False,
            'runtime': {p: importlib.metadata.version(p) for p in ['torch', 'numpy', 'pandas', 'scipy', 'anndata']},
            'samples': [{k: str(v) for k, v in s.items()} for s in samples]}
    write_json(out / 'RUN_LOCK.json', lock)
    replay = replay_original(args.data_root, models, centroids, args.device)
    write_json(out / 'original_prediction_replay.json', replay)
    print('ORIGINAL_PREDICTION_REPLAY', replay, flush=True)
    audits, metrics, associations, errors = [], [], [], []
    for i, sample in enumerate(samples, 1):
        print('START', i, len(samples), sample['sample_id'], flush=True)
        try:
            audit, rows, detection = evaluate_sample(sample, models, centroids, selected, common, ready, cal, aliases, out, args.device)
            audits.append(audit)
            metrics.extend(rows)
            associations.extend(detection)
            print('DONE', sample['sample_id'], audit['n_inferred_spots'], 'feature_coverage', audit['selected_coverage'], flush=True)
        except Exception as exc:
            errors.append({'sample_id': sample['sample_id'], 'error': repr(exc)})
            print('FAILED', sample['sample_id'], repr(exc), flush=True)
        write_json(out / 'progress.json', {'completed_samples': len(audits), 'errors': errors, 'total_samples': len(samples)})
    if metrics:
        frame = pd.DataFrame(metrics)
        frame.to_csv(out / 'section_metrics_all.csv', index=False)
        primary = frame[frame.technical_coverage_eligible].copy()
        primary.to_csv(out / 'section_metrics_coverage_eligible.csv', index=False)
        bootstrap_summary(frame, out)
        if not primary.empty:
            eligible_out = out / 'coverage_eligible_statistics'
            eligible_out.mkdir()
            bootstrap_summary(primary, eligible_out)
        pd.DataFrame(associations).to_csv(out / 'histology_associations.csv', index=False)
        pd.DataFrame([{k: v for k, v in a.items() if not isinstance(v, (dict, list))} for a in audits]).to_csv(out / 'sample_inventory.csv', index=False)
    write_json(out / 'COMPLETE.json', {'completed_utc': datetime.now(timezone.utc).isoformat(),
                                      'completed_sections': len(audits), 'failed_sections': errors,
                                      'total_spots': sum(a['n_inferred_spots'] for a in audits),
                                      'coverage_eligible_sections': sum(a['technical_coverage_eligible'] for a in audits),
                                      'model_weights_and_calibration_unchanged': True,
                                      'scope': 'retrospective frozen-joint spatial pathology proxy validation, not exact truth'})
    if errors:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
