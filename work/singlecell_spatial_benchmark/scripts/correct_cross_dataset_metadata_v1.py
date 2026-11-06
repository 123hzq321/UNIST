"""Traceable source-metadata correction; never recompute or change predictions."""
import argparse
import gzip
import json
from pathlib import Path
import shutil

import pandas as pd

from inspect_spatial_calibration_candidates_v1 import parse_samples
from validate_cross_dataset_joint_v1 import grouped_summary, group_identity, sha, write_json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', type=Path, required=True)
    args = ap.parse_args()
    root = args.run
    marker = root / 'metadata_correction.json'
    if marker.exists():
        raise FileExistsError('Metadata correction already applied')
    soft = root / 'source_metadata/GSE210041.soft.gz'
    text = gzip.decompress(soft.read_bytes()).decode('utf-8')
    if 'Formalin Fixed Paraffin Embedded (FFPE)' not in text:
        raise ValueError('Missing authoritative FFPE evidence')
    samples = parse_samples(text)
    fix = {}
    for sid in ['GSM6415705', 'GSM6415706']:
        unit, source = group_identity('GSE210041', samples[sid]['title'], '', samples[sid]['characteristics'])
        fix[sid] = {'preparation': 'FFPE', 'analysis_unit': unit, 'unit_identity_source': source}
    predictions = {p.parent.name: sha(p) for p in root.glob('*/predictions.csv.gz')}
    if len(predictions) != 42:
        raise ValueError('Correction requires the complete 42-section run')
    original = root / 'before_metadata_correction'
    original.mkdir()
    files = ['sample_inventory.csv', 'section_transport_metrics.csv', 'unit_coverage_metrics.csv',
             'cohort_coverage_summary.csv']
    before = {}
    for name in files:
        before[name] = sha(root / name)
        shutil.copyfile(root / name, original / name)
    for name in files[:2]:
        data = pd.read_csv(root / name)
        for sid, fields in fix.items():
            for field, value in fields.items():
                data.loc[data.sample_id.eq(sid), field] = value
        data.to_csv(root / name, index=False)
    grouped_summary(pd.read_csv(root / 'section_transport_metrics.csv'), root)
    for sid, expected in predictions.items():
        if sha(root / sid / 'predictions.csv.gz') != expected:
            raise ValueError('Prediction changed')
    write_json(marker, {
        'kind': 'post_run_source_metadata_correction_no_prediction_or_threshold_change',
        'source': 'https://ftp.ncbi.nlm.nih.gov/geo/series/GSE210nnn/GSE210041/soft/GSE210041_family.soft.gz',
        'source_sha256': sha(soft), 'corrections': fix,
        'reason': 'Original inventory incorrectly inferred fresh frozen and overlooked explicit patient characteristics.',
        'statistics': 'Two patients remain descriptive; no new population confidence interval or treatment contrast.',
        'original_run_lock_and_per_sample_audits_preserved': True,
        'original_table_hashes': before, 'predictions_unchanged': True,
        'prediction_hashes': predictions,
        'corrected_table_hashes': {name: sha(root / name) for name in files},
    })
    print(json.dumps({'corrected_samples': len(fix), 'predictions_unchanged': len(predictions)}))


if __name__ == '__main__':
    main()
