"""Post-run preprocessing diagnostic; does not alter frozen predictions."""
import json
import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd

from validate_frozen_joint_spatial_v1 import read_counts, discover, join_metadata, count_stream
import anndata as ad


def main():
    project = Path(__file__).resolve().parents[3]
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=Path, default=project / 'outputs/frozen_joint_spatial_20261107')
    ap.add_argument('--data-root', type=Path, default=Path(os.environ.get('UNIST_DATA_ROOT', 'external_data')))
    args = ap.parse_args()
    out, root = args.out.resolve(), args.data_root.resolve()
    if out == root or root in out.parents:
        raise ValueError('Keep outputs outside the read-only input data directory')
    aliases = json.loads((out / 'frozen_training_gene_aliases.json').read_text())['unique']
    common = json.loads((root / 'joint_training_v1/prepared/common_genes.json').read_text())
    rows = []
    for sample in discover(root):
        if sample['cohort'] == 'breast_wu2021':
            obs = pd.read_csv(sample['annotations'], index_col=0)
        else:
            a = ad.read_h5ad(sample['annotations'], backed='r')
            obs = a.obs.copy()
            a.file.close()
        x, features, barcodes = read_counts(sample['counts'])
        x, _, _, _ = join_metadata(x, barcodes, obs)
        ids = np.array([g.split('.')[0] if g.startswith('ENSG') else aliases.get(g, 'UNMAPPED:' + g)
                        for g in features[0]])
        is_gene = features[2].eq('Gene Expression').to_numpy() if 2 in features else np.ones(len(ids), bool)
        denominator = np.asarray(x[:, np.isin(ids, common) & is_gene].sum(axis=1)).ravel()
        total = np.asarray(x[:, is_gene].sum(axis=1)).ravel()
        unmatched = np.array([v.startswith('UNMAPPED:') for v in ids]) & is_gene
        unresolved_counts = np.asarray(x[:, unmatched].sum(axis=1)).ravel()
        valid = total > 0
        fraction = denominator[valid] / total[valid]
        rows.append({'cohort': sample['cohort'], 'sample_id': sample['sample_id'],
                     'n_spots': x.shape[0], 'common_gene_count_mass_fraction': float(denominator.sum() / total.sum()),
                     'unresolved_symbol_count_mass_fraction': float(unresolved_counts.sum() / total.sum()),
                     'spot_count_mass_fraction_p05': float(np.quantile(fraction, .05)),
                     'spot_count_mass_fraction_median': float(np.median(fraction)),
                     'spot_count_mass_fraction_p95': float(np.quantile(fraction, .95)),
                     'limitation': 'Only observed counts assessed; missing unmeasured transcripts remain unknowable.'})
        print(sample['sample_id'], rows[-1]['common_gene_count_mass_fraction'], flush=True)
    pd.DataFrame(rows).to_csv(out / 'normalization_count_mass_diagnostic.csv', index=False)


if __name__ == '__main__':
    main()
