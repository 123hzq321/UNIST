"""Frozen, source-held-out transport audit across four spatial studies.

Unlabelled cohorts provide coverage/shift diagnostics, not accuracy evidence.
TLS annotations provide a structure-preservation control, not mapping truth.
"""
from __future__ import annotations
import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
from datetime import datetime, timezone

import h5py
import numpy as np
import pandas as pd
import requests
from scipy import sparse
import torch

from inspect_spatial_calibration_candidates_v1 import parse_samples
from validate_frozen_joint_spatial_v1 import (
    align_counts, auc, blind_keep, load_frozen, predict, replay_original,
    score_components, sha, training_gene_aliases, write_json,
)

PROJECT = Path(__file__).resolve().parents[3]
CORE = ['entropy_only', 'novelty_only', 'disagreement_only', 'fixed_fusion']
SEED = 20261107


def decode(values):
    return [v.decode('utf-8') if isinstance(v, bytes) else str(v) for v in values]


def read_h5(path):
    with h5py.File(path, 'r') as f:
        g = f['matrix']
        shape = tuple(map(int, g['shape'][:]))
        x = sparse.csc_matrix((g['data'][:], g['indices'][:], g['indptr'][:]), shape=shape).T.tocsr().astype(np.float32)
        ids = decode(g['features/id'][:])
        names = decode(g['features/name'][:])
        kinds = decode(g['features/feature_type'][:]) if 'feature_type' in g['features'] else ['Gene Expression'] * len(ids)
        barcodes = pd.Index(decode(g['barcodes'][:]))
    if x.shape != (len(barcodes), len(ids)) or not barcodes.is_unique:
        raise ValueError('Count matrix dimensions or barcode uniqueness failed')
    if not np.isfinite(x.data).all() or np.any(x.data < 0) or not np.allclose(x.data, x.data.round()):
        raise ValueError('Expected finite nonnegative raw integer counts')
    return x, pd.DataFrame({0: ids, 1: names, 2: kinds}), barcodes


def group_identity(cohort, title, section, characteristics=None):
    if cohort == 'GSE210616':
        match = re.search(r'patient\s+(\d+)', title, re.I)
        if not match:
            raise ValueError('TNBC patient must be explicit in the GEO title, not inferred from a filename')
        return 'patient_' + match[1], 'GEO_explicit_patient_title'
    if cohort == 'GSE300676':
        match = re.fullmatch(r'CRC_(mPAP\d+)_[AB]', title)
        if not match:
            raise ValueError('Unrecognized LUAD case title')
        return match[1], 'GEO_case_title_and_four_case_overall_design'
    if cohort == 'GSE210041':
        ids = [value for value in (characteristics or []) if value.lower().startswith('patient:')]
        if len(ids) != 1 or not re.fullmatch(r'patient:\s*Patient\s+\d+', ids[0], re.I):
            raise ValueError('RCC patient must be explicit in GEO characteristics')
        return 'patient_' + ids[0].split()[-1], 'GEO_explicit_patient_characteristic_two_patient_descriptive'
    return section, 'section_only_patient_identity_unresolved'


def fetch_metadata(cohort, out):
    url = f'https://ftp.ncbi.nlm.nih.gov/geo/series/{cohort[:-3]}nnn/{cohort}/soft/{cohort}_family.soft.gz'
    response = requests.get(url, timeout=60, stream=True)
    response.raise_for_status()
    chunks, size = [], 0
    for chunk in response.iter_content(65536):
        size += len(chunk)
        if size > 5 * 1024 * 1024:
            raise ValueError('Unexpected metadata size')
        chunks.append(chunk)
    data = b''.join(chunks)
    text = gzip.decompress(data).decode('utf-8')
    (out / f'{cohort}.soft.gz').write_bytes(data)
    design = '\n'.join(x.partition(' = ')[2] for x in text.splitlines() if x.startswith('!Series_overall_design = '))
    return parse_samples(text), {'url': url, 'sha256': hashlib.sha256(data).hexdigest(), 'bytes': len(data), 'overall_design': design}


def discover(root, lung_dir, metadata):
    base = root / 'real_independent_cohorts'
    folders = {
        'GSE210616': (base / 'breast_tnbc_bassiouni_spatialge_v1/extracted/tnbc_bassiouni', 'sample_*/*filtered_feature_bc_matrix.h5'),
        'GSE175540': (base / 'renal_gse175540_rcc_tls_24samples_v1/extracted', '*filtered_feature_bc_matrix.h5'),
        'GSE210041': (base / 'renal_gse210041_ccrcc_2tumors_v1/extracted', '*filtered_feature_bc_matrix.h5'),
        'GSE300676': (lung_dir, '*filtered_feature_bc_matrix.h5'),
    }
    expected = {'GSE210616': 8, 'GSE175540': 24, 'GSE210041': 2, 'GSE300676': 8}
    rows = []
    for cohort, (folder, pattern) in folders.items():
        files = sorted(p for p in folder.glob(pattern) if not p.name.startswith('._'))
        if len(files) != expected[cohort]:
            raise ValueError(f'{cohort}: expected {expected[cohort]} local sections, found {len(files)}')
        for path in files:
            stem = path.name.removesuffix('_filtered_feature_bc_matrix.h5')
            accession, section = stem.split('_', 1)
            source = metadata[cohort][accession]
            group, group_source = group_identity(cohort, source['title'], section, source['characteristics'])
            if cohort == 'GSE175540':
                preparation = 'FFPE' if section.startswith('ffpe') else 'frozen'
            else:
                preparation = 'frozen' if cohort == 'GSE210616' else 'FFPE'
            annotation = path.with_name(stem + '_TLS_annotation.csv') if cohort == 'GSE175540' else None
            rows.append({'cohort': cohort, 'sample_id': accession, 'section_id': section,
                         'cancer': 'breast' if cohort == 'GSE210616' else ('lung' if cohort == 'GSE300676' else 'renal'),
                         'preparation': preparation, 'analysis_unit': group,
                         'unit_identity_source': group_source, 'matrix': str(path),
                         'annotation': str(annotation) if annotation and annotation.exists() else '',
                         'source_title': source['title'], 'training_relation': 'same_cancer_different_source_study',
                         'history': 'legacy_model_exposure_not_new_blinded_cohort'})
    return rows


def tls_labels(path, barcodes):
    if not path:
        return np.full(len(barcodes), np.nan), 'no_annotation', 0
    labels = pd.read_csv(path)
    if labels.Barcode.duplicated().any():
        raise ValueError('Duplicate annotation barcode')
    if 'TLS_2_cat' not in labels:
        return np.full(len(barcodes), np.nan), 'nonbinary_TLS_or_T_aggregate_excluded', 0
    values = labels.set_index('Barcode').TLS_2_cat.astype(str).str.upper()
    mapped = values.map({'TLS': 1., 'NO_TLS': 0.}).reindex(barcodes).to_numpy()
    return mapped, 'author_binary_TLS_structure_not_mapping_error', int(np.isfinite(mapped).sum())


def run_section(sample, models, centroids, selected, common, aliases, ready, cal, out, device):
    folder = out / sample['sample_id']
    folder.mkdir()
    x, features, barcodes = read_h5(sample['matrix'])
    observed_counts = np.asarray(x.sum(axis=1)).ravel()
    mapped, denominator, feature_audit = align_counts(x, features, selected, common, aliases)
    valid = denominator > 0
    x_tensor = torch.as_tensor(mapped[valid], device=device)
    barcodes = barcodes[valid]
    p, disagreement, novelty, single = predict(models, x_tensor, centroids)
    del x_tensor, mapped, x
    scores, _ = score_components(p, disagreement, novelty, cal['scales'])
    b_fraction = p[:, ready['labels'].index('B_Plasma')]
    immune = p[:, [ready['labels'].index(k) for k in ['B_Plasma', 'T_NK_ILC', 'Myeloid']]].sum(axis=1)
    y, label_role, n_labelled = tls_labels(sample['annotation'], barcodes)
    scores['composition_maxprob'] = 1 - p.max(axis=1)
    scores['composition_B_margin'] = 1 - np.abs(2 * b_fraction - 1)
    meta = {k: sample[k] for k in ['cohort', 'cancer', 'preparation', 'sample_id', 'section_id', 'analysis_unit', 'unit_identity_source']}
    coverage_eligible = feature_audit['selected_coverage'] >= .9
    audit = {**meta, **feature_audit, 'n_inferred_spots': len(p), 'n_binary_tls_annotations': n_labelled,
             'label_role': label_role, 'technical_coverage_eligible': coverage_eligible,
             'common_count_mass_fraction': float(denominator.sum() / max(observed_counts.sum(), 1)),
             'matrix_sha256': sha(sample['matrix']),
             'annotation_sha256': sha(sample['annotation']) if sample['annotation'] else None,
             'novelty_clipped_at_one_fraction': float((scores['novelty_only'] >= 1).mean()),
             'disagreement_clipped_at_one_fraction': float((scores['disagreement_only'] >= 1).mean()),
             'median_novelty_raw': float(np.median(novelty)),
             'median_disagreement_raw': float(np.median(disagreement)),
             'median_max_probability': float(np.median(p.max(axis=1)))}
    frame = pd.DataFrame({'barcode': barcodes, 'sample_id': sample['sample_id'], 'tls_label': y,
                          'novelty_raw': novelty, 'disagreement_raw': disagreement, **scores})
    for i, label in enumerate(ready['labels']):
        frame['proportion_' + label] = p[:, i]
    transport, biology = [], []
    for score in CORE:
        for q in ['0.7', '0.8', '0.9']:
            keep = scores[score] <= cal['thresholds'][score][q]
            transport.append({**meta, 'score': score, 'nominal_coverage': float(q),
                              'n_spots': len(p), 'n_retained': int(keep.sum()),
                              'actual_coverage': float(keep.mean()),
                              'coverage_minus_nominal': float(keep.mean()) - float(q),
                              'median_score': float(np.median(scores[score])),
                              'technical_coverage_eligible': coverage_eligible,
                              'interpretation': 'coverage_transport_only_not_mapping_accuracy'})
            if q == '0.8':
                frame['keep_' + score + '_frozen80'] = keep
    if np.isfinite(y).any():
        full_auc = auc(y, b_fraction)
        for score, values in scores.items():
            policies = {'matched_test_80': blind_keep(values, barcodes)}
            if score in CORE:
                policies['frozen80'] = values <= cal['thresholds'][score]['0.8']
            for policy, keep in policies.items():
                retained_auc = auc(y[keep], b_fraction[keep])
                tls = y == 1
                non_tls = y == 0
                biology.append({**meta, 'score': score, 'policy': policy,
                                'technical_coverage_eligible': coverage_eligible,
                                'n_tls': int(tls.sum()), 'n_non_tls': int(non_tls.sum()),
                                'n_tls_retained': int((tls & keep).sum()),
                                'n_non_tls_retained': int((non_tls & keep).sum()),
                                'tls_retention': float(keep[tls].mean()) if tls.any() else np.nan,
                                'non_tls_retention': float(keep[non_tls].mean()) if non_tls.any() else np.nan,
                                'tls_b_cell_fraction_auc_full': full_auc,
                                'tls_b_cell_fraction_auc_retained': retained_auc,
                                'auc_change': retained_auc - full_auc,
                                'tls_immune_fraction_auc_full_secondary': auc(y, immune),
                                'uncertainty_tls_association_auc': auc(y, values),
                                'interpretation': 'TLS_structure_preservation_control_not_composition_truth'})
                frame['keep_' + score + '_' + policy] = keep
    frame.to_csv(folder / 'predictions.csv.gz', index=False)
    write_json(folder / 'input_audit.json', audit)
    return audit, transport, biology


def grouped_summary(transport, out):
    group_cols = ['cohort', 'cancer', 'preparation', 'score', 'nominal_coverage', 'unit_identity_source', 'analysis_unit']
    units = transport.groupby(group_cols, as_index=False).agg(actual_coverage=('actual_coverage', 'mean'),
                                                             n_sections=('sample_id', 'nunique'))
    units.to_csv(out / 'unit_coverage_metrics.csv', index=False)
    rows = []
    for keys, group in units.groupby(group_cols[:-1]):
        values = group.actual_coverage.to_numpy()
        verified = not keys[-1].startswith('section_only') and 'two_patient_descriptive' not in keys[-1]
        low, high = np.nan, np.nan
        if verified and len(values) > 1:
            rng = np.random.default_rng(SEED)
            draws = rng.choice(values, (5000, len(values)), replace=True).mean(axis=1)
            low, high = np.quantile(draws, [.025, .975])
        rows.append(dict(zip(group_cols[:-1], keys), mean_coverage=float(values.mean()),
                         n_units=len(values), n_sections=int(group.n_sections.sum()),
                         ci_low=low, ci_high=high,
                         inference='within_source_patient_or_case_bootstrap' if verified else 'section_descriptive_no_population_CI'))
    pd.DataFrame(rows).to_csv(out / 'cohort_coverage_summary.csv', index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-root', type=Path, default=Path(os.environ.get('UNIST_DATA_ROOT', 'external_data')))
    ap.add_argument('--lung-dir', type=Path, required=True)
    ap.add_argument('--out', type=Path, default=PROJECT / 'outputs/cross_dataset_joint_20261107')
    ap.add_argument('--previous-audit', type=Path)
    ap.add_argument('--device', default='cuda')
    args = ap.parse_args()
    root, out = args.data_root.resolve(), args.out.resolve()
    if root == out or root in out.parents or out.drive.upper() == 'D:':
        raise ValueError('Use a separate healthy output volume; D: inputs are read-only in this recovery run')
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'RUN_LOCK.json').exists():
        raise FileExistsError('Do not overwrite a locked study run')
    protocol = PROJECT / 'studies/cross_dataset_joint_20261107/PROTOCOL.txt'
    shutil.copyfile(protocol, out / 'PROTOCOL.txt')
    meta_dir = out / 'source_metadata'
    meta_dir.mkdir()
    metadata, provenance = {}, {}
    for cohort in ['GSE175540', 'GSE210616', 'GSE210041', 'GSE300676']:
        metadata[cohort], provenance[cohort] = fetch_metadata(cohort, meta_dir)
    samples = discover(root, args.lung_dir.resolve(), metadata)
    torch.set_num_threads(4)
    models, centroids, selected, common, ready, cal, integrity = load_frozen(root, args.device)
    aliases, _ = training_gene_aliases(root)
    protected = []
    if args.previous_audit:
        for file in args.previous_audit.glob('*/predictions.csv.gz'):
            protected.append({'path': str(file), 'sha256': sha(file)})
        if len(protected) != 14:
            raise ValueError('Expected the previous 14-section audit, not a partial subset')
    lock = {'started_utc': datetime.now(timezone.utc).isoformat(), 'script_sha256': sha(__file__),
            'protocol_sha256': sha(protocol), 'sources': provenance, 'samples': samples,
            'model_integrity': integrity, 'frozen_calibration': cal, 'protected_previous_predictions': protected,
            'label_use': 'no tuning; TLS is structure control, no exact error truth',
            'no_training_or_threshold_refit': True, 'source_studies_held_out_from_joint_fit': True,
            'cross_study_patient_aliases': 'not definitively resolved',
            'not_a_new_blinded_cohort': True, 'expected_sections': 42}
    write_json(out / 'RUN_LOCK.json', lock)
    write_json(out / 'original_prediction_replay.json', replay_original(root, models, centroids, args.device))
    audits, transports, biology, errors = [], [], [], []
    for i, sample in enumerate(samples, 1):
        print('START', i, len(samples), sample['cohort'], sample['sample_id'], flush=True)
        try:
            audit, rows, bio = run_section(sample, models, centroids, selected, common, aliases, ready, cal, out, args.device)
            audits.append(audit)
            transports.extend(rows)
            biology.extend(bio)
            print('DONE', sample['sample_id'], audit['n_inferred_spots'], 'coverage', audit['selected_coverage'], flush=True)
        except Exception as exc:
            errors.append({'sample_id': sample['sample_id'], 'error': repr(exc)})
            print('FAILED', sample['sample_id'], repr(exc), flush=True)
        write_json(out / 'progress.json', {'completed_sections': len(audits), 'errors': errors, 'expected': 42})
    inventory = pd.DataFrame([{k: v for k, v in a.items() if not isinstance(v, (dict, list))} for a in audits])
    inventory.to_csv(out / 'sample_inventory.csv', index=False)
    transport = pd.DataFrame(transports)
    transport.to_csv(out / 'section_transport_metrics.csv', index=False)
    pd.DataFrame(biology).to_csv(out / 'tls_structure_metrics.csv', index=False)
    grouped_summary(transport, out)
    for artifact in integrity + protected:
        if sha(artifact['path']) != artifact['sha256']:
            raise ValueError('Protected artifact changed: ' + artifact['path'])
    write_json(out / 'COMPLETE.json', {'completed_utc': datetime.now(timezone.utc).isoformat(),
                                      'cohorts_completed': int(inventory.cohort.nunique()),
                                      'sections_completed': len(audits), 'spots': int(inventory.n_inferred_spots.sum()),
                                      'coverage_eligible_sections': int(inventory.technical_coverage_eligible.sum()),
                                      'sections_with_binary_tls_labels': int((inventory.n_binary_tls_annotations > 0).sum()),
                                      'errors': errors, 'protected_artifacts_unchanged': True,
                                      'no_new_training': True, 'no_accuracy_claim_for_unlabelled_queries': True})
    if errors:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
