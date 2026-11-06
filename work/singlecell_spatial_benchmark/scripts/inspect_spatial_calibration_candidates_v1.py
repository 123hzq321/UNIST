"""Read-only candidate inspection; never invent patient-level split identities."""
import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import re

import h5py
import pandas as pd
import requests

URL = 'https://ftp.ncbi.nlm.nih.gov/geo/series/GSE175nnn/GSE175540/soft/GSE175540_family.soft.gz'
PROJECT = Path(__file__).resolve().parents[3]


def parse_samples(text):
    result, current = {}, None
    for line in text.splitlines():
        key, sep, value = line.partition(' = ')
        if not sep:
            continue
        if key == '^SAMPLE':
            current = value
            result[current] = {'characteristics': []}
        elif current and key == '!Sample_title':
            result[current]['title'] = value
        elif current and key == '!Sample_characteristics_ch1':
            result[current]['characteristics'].append(value)
    return result


def decode(values):
    return [v.decode('utf-8') if isinstance(v, bytes) else str(v) for v in values]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-root', type=Path, default=Path(os.environ.get('UNIST_DATA_ROOT', 'external_data')))
    ap.add_argument('--out', type=Path, default=PROJECT / 'outputs/spatial_calibration_candidate_audit_20261107')
    args = ap.parse_args()
    out, root = args.out.resolve(), args.data_root.resolve()
    if out == root or root in out.parents:
        raise ValueError('Do not write candidate outputs under the input data directory')
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'candidate_inventory.csv').exists():
        raise FileExistsError('Candidate inspection already exists; do not replace its provenance')
    response = requests.get(URL, timeout=40, stream=True)
    response.raise_for_status()
    parts, total = [], 0
    for part in response.iter_content(65536):
        total += len(part)
        if total > 5 * 1024 * 1024:
            raise ValueError('Unexpectedly large metadata response')
        parts.append(part)
    content = b''.join(parts)
    metadata = gzip.decompress(content).decode('utf-8')
    samples = parse_samples(metadata)
    (out / 'GSE175540_metadata_source.soft.gz').write_bytes(content)
    selected = set(json.loads((root / 'joint_training_v1/prepared/selected_genes.json').read_text()))
    source = root / 'real_independent_cohorts/renal_gse175540_rcc_tls_24samples_v1/extracted'
    rows = []
    for path in sorted(source.glob('*filtered_feature_bc_matrix.h5')):
        stem = path.name.removesuffix('_filtered_feature_bc_matrix.h5')
        accession, sample = stem.split('_', 1)
        if accession not in samples:
            raise ValueError(f'Local sample missing from authoritative metadata: {accession}')
        with h5py.File(path, 'r') as f:
            features = {g.split('.')[0] for g in decode(f['matrix/features/id'][:])}
            barcodes = set(decode(f['matrix/barcodes'][:]))
        annotation = source / (stem + '_TLS_annotation.csv')
        n_annotated, n_tls, duplicated, n_binary = 0, 0, 0, 0
        annotation_column, categories = '', {}
        if annotation.exists():
            labels = pd.read_csv(annotation)
            duplicated = int(labels.Barcode.duplicated().sum())
            labels = labels[labels.Barcode.isin(barcodes)]
            n_annotated = len(labels)
            annotation_column = 'TLS_2_cat' if 'TLS_2_cat' in labels else 'TLS'
            values = labels[annotation_column].fillna('UNSPECIFIED').astype(str).str.upper()
            categories = {k: int(v) for k, v in values.value_counts().items()}
            n_tls = int(values.eq('TLS').sum())
            if annotation_column == 'TLS_2_cat':
                n_binary = int(values.isin(['TLS', 'NO_TLS']).sum())
        identity = [v for v in samples[accession]['characteristics']
                    if re.match(r'(patient|donor|subject)([ _-]?(id|identifier|number))?\s*:', v, re.I)]
        rows.append({'accession': accession, 'section_id': sample,
                     'preparation_from_author_title': 'FFPE' if sample.startswith('ffpe') else 'frozen',
                     'n_spots_in_expression': len(barcodes), 'n_spots_with_tls_annotation': n_annotated,
                     'n_tls_spots': n_tls, 'duplicate_annotation_barcodes': duplicated,
                     'annotation_column': annotation_column,
                     'annotation_categories': json.dumps(categories, sort_keys=True),
                     'n_explicit_binary_tls_annotations': n_binary,
                     'n_model_features_present': len(features & selected),
                     'model_feature_coverage': len(features & selected) / len(selected),
                     'explicit_patient_or_donor_field_present': bool(identity),
                     'split_role': 'unassigned_pending_identity_verification',
                     'label_role': 'author_TLS_structure_annotation_not_mapping_error_truth',
                     'prior_use': 'legacy_models_already_evaluated_not_a_new_blinded_cohort'})
    frame = pd.DataFrame(rows)
    frame.to_csv(out / 'candidate_inventory.csv', index=False)
    summary = {'source_url': URL, 'metadata_sha256': hashlib.sha256(content).hexdigest(),
               'metadata_download_bytes': len(content), 'geo_samples': len(samples),
               'local_sections': len(frame), 'local_spots': int(frame.n_spots_in_expression.sum()),
               'sections_with_annotations': int((frame.n_spots_with_tls_annotation > 0).sum()),
               'sections_with_explicit_binary_tls_annotations': int((frame.n_explicit_binary_tls_annotations > 0).sum()),
               'sections_with_explicit_identity_metadata': int(frame.explicit_patient_or_donor_field_present.sum()),
               'new_model_predictions_run': False, 'new_patient_level_split_created': False,
               'original_14_section_validation_unchanged': True, 'input_data_written': False}
    (out / 'candidate_inspection.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    lines = ['Spatial calibration candidate audit - 7 November 2026', '',
             'GSE175540', f'Authoritative GEO records: {len(samples)}; local sections: {len(frame)}.',
             f"Expression spots: {summary['local_spots']:,}; sections with matched TLS annotations: {summary['sections_with_annotations']}.",
             f"Sections with explicit TLS/NO_TLS annotations: {summary['sections_with_explicit_binary_tls_annotations']}.",
             'Other files use a TLS column with T_AGG or unspecified values. They are not',
             'silently converted to binary TLS ground truth. One section has no annotation file.',
             'Feature coverage by preparation:',
             frame.groupby('preparation_from_author_title').agg(
                 sections=('section_id', 'size'), min_coverage=('model_feature_coverage', 'min'),
                 max_coverage=('model_feature_coverage', 'max')).to_string(), '',
             'No explicit patient/donor identity fields were found in the retrieved GEO',
             'sample characteristics. A section name is not a verified patient ID.',
             'Some author section names recur across preparation types (for example c_2);',
             'do not independently split such sections without checking primary clinical metadata.',
             'No patient split, new prediction, calibration fit or efficacy claim was produced.',
             'This is a candidate inspection, not an additional completed validation cohort.', '',
             'TNBC Zenodo 8135722 provenance warning',
             'The author README describes classifAll.RDS as classifications produced by',
             'regressors, with baseClassif.RDS used for gene-level rotation/scaling.',
             'These are not direct pathologist labels. For orthogonal validation, trace',
             'annotBySpot/artefacts annotations and the array-to-sample/patient mapping.',
             'Do not inherit the older local hard_gold ledger label without that check.', '',
             'Required next inputs',
             '1. Verified section-to-patient identities from the original study supplement.',
             '2. A genuinely independent development/calibration/test assignment by patient.',
             '3. Explicitly separate TLS biology controls from composition-error truth.',
             '4. Retain the previously observed 14 sections as audit/failure data, not tuning data.', '',
             'Source: ' + URL,
             'TNBC author repository: https://zenodo.org/records/8135722']
    (out / 'REPORT.txt').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
