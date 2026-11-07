"""Calibration-only spatial coverage policies; never a mapping-error guarantee."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
from datetime import datetime, timezone

import h5py
import numpy as np
import pandas as pd
from scipy import sparse
import torch

from validate_cross_dataset_joint_v1 import decode, fetch_metadata
from validate_frozen_joint_spatial_v1 import (
    CORE, align_counts, auc, load_frozen, predict, replay_original, score_components,
    sha, training_gene_aliases, write_json,
)

PROJECT = Path(__file__).resolve().parents[3]
SCORES = CORE + ['novelty_raw', 'disagreement_raw', 'composition_maxprob']
TARGET = .8
SEED = 20261107


def weights(frame):
    units = frame.analysis_unit.nunique()
    per_unit_sections = frame.groupby('analysis_unit').sample_id.transform('nunique')
    per_section_spots = frame.groupby(['analysis_unit', 'sample_id']).sample_id.transform('size')
    return (1 / (units * per_unit_sections * per_section_spots)).to_numpy()


def fit_cutoff(values, mass, q=TARGET):
    values, mass = np.asarray(values, float), np.asarray(mass, float)
    if not (len(values) and len(values) == len(mass) and np.isfinite(values).all()
            and np.isfinite(mass).all() and (mass > 0).all() and 0 < q < 1):
        raise ValueError('Invalid calibration values, weights or target')
    order = np.argsort(values, kind='stable')
    v, w = values[order], mass[order] / mass.sum()
    cutoff = v[min(int(np.searchsorted(np.cumsum(w), q, side='left')), len(v) - 1)]
    below = float(w[v < cutoff].sum())
    tied = float(w[v == cutoff].sum())
    return {'value': float(cutoff), 'tie_acceptance': float(np.clip((q - below) / tied, 0, 1)),
            'below_mass': below, 'tied_mass': tied, 'expected_calibration_coverage': q}


def hash_uniform(sample, barcodes):
    return np.array([int.from_bytes(hashlib.sha256(f'{SEED}|{sample}|{b}'.encode()).digest()[:8], 'big')
                     / 2**64 for b in barcodes])


def apply_cutoff(values, uniform, cutoff):
    v = np.asarray(values, float)
    return (v < cutoff['value']) | ((v == cutoff['value']) & (uniform < cutoff['tie_acceptance']))


def augment(frame):
    frame = frame.copy()
    columns = sorted(c for c in frame if c.startswith('proportion_'))
    if len(columns) != 7:
        raise ValueError('Expected the seven original composition labels')
    p = frame[columns].to_numpy()
    if not np.isfinite(p).all() or not np.allclose(p.sum(axis=1), 1, atol=1e-5):
        raise ValueError('Invalid original predictions')
    frame['dominant'] = np.array(columns)[p.argmax(axis=1)]
    frame['composition_maxprob'] = 1 - p.max(axis=1)
    return frame


def fit_policies(frame):
    result = {}
    for prep, group in frame.groupby('preparation'):
        rules = {}
        for score in SCORES:
            pooled = fit_cutoff(group[score], weights(group))
            bins = {}
            for label, subset in group.groupby('dominant'):
                supported = len(subset) >= 100 and subset.analysis_unit.nunique() >= 3
                bins[label] = {'rule': fit_cutoff(subset[score], weights(subset)) if supported else pooled,
                               'fallback_to_pooled': not supported,
                               'n_spots': len(subset), 'n_units': subset.analysis_unit.nunique()}
            rules[score] = {'pooled': pooled, 'conditional': bins,
                            'calibration_constant': bool(group[score].nunique() == 1)}
        result[prep] = rules
    return result


def conditional_keep(frame, score, uniform, rule):
    keep = np.zeros(len(frame), dtype=bool)
    for label in frame.dominant.unique():
        mask = frame.dominant.eq(label).to_numpy()
        cutoff = rule['conditional'].get(label, {'rule': rule['pooled']})['rule']
        keep[mask] = apply_cutoff(frame.loc[mask, score], uniform[mask], cutoff)
    return keep


def evaluate(frame, meta, policies, original, eligible, role):
    if meta['preparation'] not in policies:
        raise ValueError('No preparation-matched calibration; do not extrapolate')
    uniform = hash_uniform(meta['sample_id'], frame.barcode)
    sets = []
    for score in SCORES:
        values = frame[score].to_numpy()
        rule = policies[meta['preparation']][score]
        sets += [(score, 'spatial_pooled', apply_cutoff(values, uniform, rule['pooled'])),
                 (score, 'spatial_class_conditional', conditional_keep(frame, score, uniform, rule))]
        if score in CORE:
            sets.append((score, 'original_synthetic', values <= original['thresholds'][score]['0.8']))
    sets.append(('label_blind_hash', 'null_control', uniform < TARGET))
    rows, biology = [], []
    y = frame.tls_label.to_numpy() if 'tls_label' in frame else np.full(len(frame), np.nan)
    tls, other = y == 1, y == 0
    for score, policy, keep in sets:
        values = frame[score].to_numpy() if score != 'label_blind_hash' else uniform
        rows.append({**meta, 'role': role, 'score': score, 'policy': policy,
                     'n_spots': len(frame), 'n_retained': int(keep.sum()),
                     'hypothetical_coverage': float(keep.mean()),
                     'coverage_minus_target': float(keep.mean() - TARGET),
                     'input_coverage_eligible': eligible,
                     'status': 'coverage_only_not_accuracy_certified' if eligible else 'review_only_feature_limited',
                     'gate_allowed_retained': int(keep.sum()) if eligible else 0,
                     'score_unique_values': int(len(np.unique(values))),
                     'largest_tie_fraction': float(pd.Series(values).value_counts().iloc[0] / len(values))})
        if tls.any() and other.any():
            b = frame.proportion_B_Plasma.to_numpy()
            biology.append({**meta, 'role': role, 'score': score, 'policy': policy,
                            'n_tls': int(tls.sum()), 'n_non_tls': int(other.sum()),
                            'tls_retention': float(keep[tls].mean()),
                            'non_tls_retention': float(keep[other].mean()),
                            'tls_minus_non_tls': float(keep[tls].mean() - keep[other].mean()),
                            'b_tls_auc_full': auc(y, b), 'b_tls_auc_retained': auc(y[keep], b[keep]),
                            'input_coverage_eligible': eligible,
                            'interpretation': 'retrospective_structure_diagnostic_not_independent_rescue'})
    return rows, biology


def read_h5_metadata(path):
    with h5py.File(path, 'r') as f:
        g = f['matrix']
        features = pd.DataFrame({0: decode(g['features/id'][:]), 1: decode(g['features/name'][:]),
                                 2: decode(g['features/feature_type'][:])})
        barcodes = pd.Index(decode(g['barcodes'][:]))
        shape = tuple(map(int, g['shape'][:]))
    if shape != (len(features), len(barcodes)) or not barcodes.is_unique:
        raise ValueError('Invalid 10x shape or duplicate barcodes')
    return features, barcodes, shape


def infer_h5(path, models, centroids, selected, common, aliases, ready, original, device, chunk=8192):
    path = Path(path)
    features, barcodes, shape = read_h5_metadata(path)
    pieces, denominator_mass, observed_mass, removed = [], 0., 0., 0
    with h5py.File(path, 'r') as f:
        g = f['matrix']
        ptr = g['indptr'][:]
        for start in range(0, shape[1], chunk):
            end = min(start + chunk, shape[1])
            left, right = int(ptr[start]), int(ptr[end])
            x = sparse.csc_matrix((g['data'][left:right], g['indices'][left:right],
                                   ptr[start:end + 1] - left), shape=(shape[0], end-start)).T.tocsr().astype(np.float32)
            if not np.isfinite(x.data).all() or (x.data < 0).any() or not np.allclose(x.data, x.data.round()):
                raise ValueError('Counts must be nonnegative finite integers')
            mapped, denom, audit = align_counts(x, features, selected, common, aliases)
            valid = denom > 0
            denominator_mass += float(denom.sum())
            observed_mass += float(np.asarray(x.sum(axis=1), dtype=np.float64).sum())
            removed += int((~valid).sum())
            if valid.any():
                tensor = torch.as_tensor(mapped[valid], device=device)
                p, d, n, _ = predict(models, tensor, centroids)
                del tensor
                scores, _ = score_components(p, d, n, original['scales'])
                frame = pd.DataFrame({'barcode': barcodes[start:end][valid], 'novelty_raw': n,
                                      'disagreement_raw': d, **scores})
                for j, label in enumerate(ready['labels']):
                    frame['proportion_' + label] = p[:, j]
                pieces.append(frame)
            if shape[1] > 50000 and (start // chunk) % 20 == 0:
                print('CHUNK', path.name, end, shape[1], flush=True)
    if not pieces:
        raise ValueError('No nonzero-common-count spots')
    audit['zero_common_count_spots'] = removed
    audit['input_spots'] = shape[1]
    audit['common_count_mass_fraction'] = denominator_mass / max(observed_mass, 1)
    return augment(pd.concat(pieces, ignore_index=True)), audit


def preparation_from_soft(path, accession):
    import gzip
    text = gzip.decompress(path.read_bytes()).decode('utf-8')
    body = text.split('^SAMPLE = ' + accession, 1)[1].split('^SAMPLE = ', 1)[0]
    protocol = ' '.join(line for line in body.splitlines() if line.startswith('!Sample_extract_protocol_ch1'))
    if re.search(r'FFPE|paraffin', protocol, re.I):
        return 'FFPE'
    if re.search(r'\bOCT\b|fresh.frozen|snap.frozen|cryosection|frozen', protocol, re.I):
        return 'frozen'
    raise ValueError(f'Preparation not resolved from {accession} source protocol')


def discover(root, out):
    definitions = {
        'GSE251950': list((root / 'gse251950_gastric_visium_v1/extracted').glob('*/*/*filtered_feature_bc_matrix.h5')),
        'GSE250163': list((root / 'renal_second_cohort_gse250163/GSM7974888').glob('*filtered_feature_bc_matrix.h5')),
        'GSE280318': [root / 'st_he_pathology/visium_hd_crc_p2/GSM8594568_P2CRC_filtered_feature_bc_matrix.h5',
                     root / 'real_independent_cohorts/crc_visium_hd_p5_gsm8594569/GSM8594569_P5CRC_filtered_feature_bc_matrix.h5'],
    }
    expected = {'GSE251950': 10, 'GSE250163': 1, 'GSE280318': 2}
    meta_dir = out / 'source_metadata'
    meta_dir.mkdir()
    rows, sources = [], {}
    for cohort, files in definitions.items():
        if len(files) != expected[cohort] or not all(p.exists() for p in files):
            raise ValueError('Missing new source inputs: ' + cohort)
        samples, sources[cohort] = fetch_metadata(cohort, meta_dir)
        for path in sorted(files):
            if cohort == 'GSE251950':
                accession = path.parents[1].name.split('_')[0]
            else:
                accession = path.name.split('_')[0]
            source = samples[accession]
            rows.append({'cohort': cohort, 'sample_id': accession, 'source_title': source['title'],
                         'preparation': preparation_from_soft(meta_dir / (cohort + '.soft.gz'), accession),
                         'cancer': {'GSE251950': 'gastric', 'GSE250163': 'renal', 'GSE280318': 'colorectal'}[cohort],
                         'analysis_unit': accession, 'identity': 'section_descriptive_no_patient_inference',
                         'matrix': str(path), 'matrix_sha256': sha(path),
                         'relation': 'new_joint_model_test_legacy_model_exposed_no_exact_truth'})
    if len({r['matrix_sha256'] for r in rows}) != 13:
        raise ValueError('Duplicate test inputs')
    return rows, sources


def resume_failed(args):
    out, root = args.out.resolve(), args.data_root.resolve()
    if out.drive.upper() == 'D:' or root == out or root in out.parents:
        raise ValueError('Input volume remains read-only')
    lock = json.loads((out / 'RUN_LOCK.json').read_text())
    previous = json.loads((out / 'COMPLETE.json').read_text())
    if not previous['errors']:
        raise ValueError('No failed test items to resume')
    if sha(out / 'fitted_policies.json') != lock['policy_sha256']:
        raise ValueError('Frozen policy changed')
    policies = json.loads((out / 'fitted_policies.json').read_text())
    archive = out / 'attempt_001_failed'
    archive.mkdir()
    for name in ['COMPLETE.json', 'progress.json', 'section_policy_metrics.csv', 'new_source_inventory.csv']:
        shutil.copyfile(out / name, archive / name)
    metrics = pd.read_csv(out / 'section_policy_metrics.csv').to_dict('records')
    audits = pd.read_csv(out / 'new_source_inventory.csv').to_dict('records')
    failed = {row['sample_id'] for row in previous['errors']}
    samples = [sample for sample in lock['test_samples'] if sample['sample_id'] in failed]
    if any(a['sample_id'] in failed for a in audits):
        raise ValueError('Retry would duplicate a completed sample')
    torch.set_num_threads(4)
    models, centroids, selected, common, ready, original, integrity = load_frozen(root, args.device)
    aliases, _ = training_gene_aliases(root)
    errors = []
    for sample in samples:
        print('RETRY START', sample['sample_id'], flush=True)
        if sha(sample['matrix']) != sample['matrix_sha256']:
            raise ValueError('Test source changed')
        try:
            frame, audit = infer_h5(sample['matrix'], models, centroids, selected, common, aliases, ready, original, args.device)
            meta = {k: sample[k] for k in ['sample_id', 'cohort', 'cancer', 'preparation', 'analysis_unit']}
            eligible = audit['selected_coverage'] >= .9
            rows, _ = evaluate(frame, meta, policies, original, eligible, 'new_source_joint_model_test')
            metrics.extend(rows)
            audits.append({**meta, **{k: v for k, v in audit.items() if not isinstance(v, list)},
                           'n_spots': len(frame), 'input_coverage_eligible': eligible,
                           'matrix_sha256': sample['matrix_sha256'],
                           'novelty_clipped_fraction': float(frame.novelty_only.eq(1).mean())})
            print('RETRY DONE', sample['sample_id'], len(frame), flush=True)
            del frame
        except Exception as exc:
            errors.append({'sample_id': sample['sample_id'], 'error': repr(exc)})
            print('RETRY FAILED', sample['sample_id'], repr(exc), flush=True)
        write_json(out / 'progress.json', {'completed': len(audits), 'expected': 13, 'errors': errors})
    for item in lock['model_integrity'] + lock['calibration_prediction_hashes']:
        if sha(item['path']) != item['sha256']:
            raise ValueError('Protected input changed')
    if sha(out / 'fitted_policies.json') != lock['policy_sha256'] or sha(out / 'PROTOCOL.txt') != lock['protocol_sha256']:
        raise ValueError('Policy or protocol changed during retry')
    pd.DataFrame(metrics).to_csv(out / 'section_policy_metrics.csv', index=False)
    pd.DataFrame(audits).to_csv(out / 'new_source_inventory.csv', index=False)
    write_json(out / 'resume_record.json', {
        'reason': 'Normalize path to Path before progress logging in infer_h5; no numerical or policy change.',
        'original_script_sha256': lock['script_sha256'], 'resume_script_sha256': sha(__file__),
        'retried_samples': sorted(failed), 'policy_sha256': lock['policy_sha256'],
        'previous_completed_sections_not_rerun': previous['new_sections'],
        'no_refit': True, 'original_failure_artifacts_preserved': True})
    write_json(out / 'COMPLETE.json', {'completed_utc': datetime.now(timezone.utc).isoformat(),
                                      'new_sections': len(audits), 'new_spots': int(sum(a['n_spots'] for a in audits)),
                                      'new_eligible_sections': int(sum(a['input_coverage_eligible'] for a in audits)),
                                      'calibration_sections': 16, 'retrospective_tls_sections': 24,
                                      'errors': errors, 'policies_and_protected_artifacts_unchanged': True,
                                      'no_mapper_retraining': True})
    if errors:
        raise SystemExit(2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-root', type=Path, required=True)
    ap.add_argument('--previous-run', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--resume-failed', action='store_true')
    args = ap.parse_args()
    if args.resume_failed:
        resume_failed(args)
        return
    out = args.out.resolve()
    root = args.data_root.resolve()
    if out.drive.upper() == 'D:' or out == root or root in out.parents:
        raise ValueError('Input D: is read-only; use a healthy recovery output volume')
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'RUN_LOCK.json').exists():
        raise FileExistsError('Do not overwrite a locked policy experiment')
    protocol = PROJECT / 'studies/spatial_policy_transfer_20261107/PROTOCOL.txt'
    shutil.copyfile(protocol, out / 'PROTOCOL.txt')
    inventory = pd.read_csv(args.previous_run / 'sample_inventory.csv')
    cal_inventory = inventory[inventory.cohort.isin(['GSE210616', 'GSE300676'])]
    if len(cal_inventory) != 16 or cal_inventory.groupby('cohort').analysis_unit.nunique().tolist() != [4, 4]:
        raise ValueError('Calibration source identity mismatch')
    calibration, protected = [], []
    for sample in cal_inventory.to_dict('records'):
        path = args.previous_run / sample['sample_id'] / 'predictions.csv.gz'
        protected.append({'path': str(path), 'sha256': sha(path)})
        frame = augment(pd.read_csv(path))
        for col in ['cohort', 'sample_id', 'analysis_unit', 'preparation']:
            frame[col] = sample[col]
        calibration.append(frame)
    combined = pd.concat(calibration, ignore_index=True)
    policies = fit_policies(combined)
    write_json(out / 'fitted_policies.json', policies)
    torch.set_num_threads(4)
    models, centroids, selected, common, ready, original, integrity = load_frozen(root, args.device)
    aliases, _ = training_gene_aliases(root)
    samples, sources = discover(root, out)
    previous_hashes = set(inventory.matrix_sha256)
    if previous_hashes & {r['matrix_sha256'] for r in samples}:
        raise ValueError('Test matrix duplicates a previous input')
    lock = {'started_utc': datetime.now(timezone.utc).isoformat(), 'protocol_sha256': sha(protocol),
            'script_sha256': sha(__file__), 'policy_sha256': sha(out / 'fitted_policies.json'),
            'model_integrity': integrity, 'calibration_prediction_hashes': protected,
            'test_samples': samples, 'sources': sources,
            'calibration_studies': ['GSE210616', 'GSE300676'],
            'test_studies': ['GSE251950', 'GSE250163', 'GSE280318'],
            'test_patients_verified': False, 'new_cancer_categories': ['gastric', 'colorectal'],
            'no_mapper_retraining': True, 'no_risk_or_conformal_guarantee': True,
            'retrospective_tls_diagnostic': 'GSE175540; known preceding outcomes; no fitting to TLS labels'}
    write_json(out / 'RUN_LOCK.json', lock)
    write_json(out / 'original_prediction_replay.json', replay_original(root, models, centroids, args.device))
    metrics, bio, audits, errors = [], [], [], []
    for (sid, prep), frame in combined.groupby(['sample_id', 'preparation']):
        source = cal_inventory[cal_inventory.sample_id.eq(sid)].iloc[0]
        meta = {k: source[k] for k in ['sample_id', 'cohort', 'cancer', 'preparation', 'analysis_unit']}
        rows, _ = evaluate(frame, meta, policies, original, bool(source.technical_coverage_eligible), 'calibration_in_sample_not_test')
        metrics.extend(rows)
    for i, sample in enumerate(samples, 1):
        print('TEST START', i, 13, sample['sample_id'], sample['preparation'], flush=True)
        try:
            frame, audit = infer_h5(sample['matrix'], models, centroids, selected, common, aliases, ready, original, args.device)
            meta = {k: sample[k] for k in ['sample_id', 'cohort', 'cancer', 'preparation', 'analysis_unit']}
            eligible = audit['selected_coverage'] >= .9
            rows, _ = evaluate(frame, meta, policies, original, eligible, 'new_source_joint_model_test')
            metrics.extend(rows)
            audits.append({**meta, **audit, 'n_spots': len(frame), 'input_coverage_eligible': eligible,
                           'matrix_sha256': sample['matrix_sha256'],
                           'novelty_clipped_fraction': float(frame.novelty_only.eq(1).mean())})
            print('TEST DONE', sample['sample_id'], len(frame), audit['selected_coverage'], flush=True)
            del frame
        except Exception as exc:
            errors.append({'sample_id': sample['sample_id'], 'error': repr(exc)})
            print('TEST FAILED', sample['sample_id'], repr(exc), flush=True)
        write_json(out / 'progress.json', {'completed': len(audits), 'expected': 13, 'errors': errors})
    for sample in inventory[inventory.cohort.eq('GSE175540')].to_dict('records'):
        frame = augment(pd.read_csv(args.previous_run / sample['sample_id'] / 'predictions.csv.gz'))
        meta = {k: sample[k] for k in ['sample_id', 'cohort', 'cancer', 'preparation', 'analysis_unit']}
        rows, biology = evaluate(frame, meta, policies, original, bool(sample['technical_coverage_eligible']), 'retrospective_tls_diagnostic')
        metrics.extend(rows)
        bio.extend(biology)
    pd.DataFrame(metrics).to_csv(out / 'section_policy_metrics.csv', index=False)
    pd.DataFrame(bio).to_csv(out / 'retrospective_tls_metrics.csv', index=False)
    pd.DataFrame([{k: v for k, v in a.items() if not isinstance(v, list)} for a in audits]).to_csv(out / 'new_source_inventory.csv', index=False)
    for item in integrity + protected:
        if sha(item['path']) != item['sha256']:
            raise ValueError('Protected source artifact changed')
    if sha(out / 'fitted_policies.json') != lock['policy_sha256'] or sha(protocol) != lock['protocol_sha256']:
        raise ValueError('Policies or protocol changed during test')
    write_json(out / 'COMPLETE.json', {'completed_utc': datetime.now(timezone.utc).isoformat(),
                                      'new_sections': len(audits), 'new_spots': sum(a['n_spots'] for a in audits),
                                      'new_eligible_sections': sum(a['input_coverage_eligible'] for a in audits),
                                      'calibration_sections': 16, 'retrospective_tls_sections': 24,
                                      'errors': errors, 'policies_and_protected_artifacts_unchanged': True,
                                      'no_mapper_retraining': True})
    if errors:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
