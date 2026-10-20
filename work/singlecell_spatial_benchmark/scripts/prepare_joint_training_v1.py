"""Patient-disjoint, train-only feature preparation for joint reference training."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import h5py
import numpy as np
import pandas as pd
from anndata.io import read_elem, sparse_dataset

from joint_paths import joint_root

ROOT = joint_root()
SEED = 20260926

def donor_split(donors, source, seed=SEED):
    donors = sorted(set(map(str, donors)))
    if len(donors) < 5:
        raise ValueError('At least five donors required: '+source)
    source_seed = int(hashlib.sha256((source+str(seed)).encode()).hexdigest()[:8], 16)
    order = np.random.default_rng(source_seed).permutation(donors)
    n = max(1, int(round(len(donors)*0.15)))
    return {d: ('test' if i < n else 'calibration' if i < 2*n else 'train')
            for i,d in enumerate(order)}

def matrix_info(f, rec):
    prefix = 'raw/' if rec['raw_data_location'] == 'raw.X' else ''
    var = read_elem(f[prefix+'var'])
    genes = var.index.astype(str).str.replace(r'\.\d+$', '', regex=True)
    if genes.duplicated().any():
        raise ValueError('Duplicate gene identifiers: '+rec['cancer'])
    return sparse_dataset(f[prefix+'X']), var, genes

def count_block(reader, start, stop, columns):
    block = reader[start:stop][:, columns].tocsr().astype(np.float32)
    if np.any(~np.isfinite(block.data)) or np.any(block.data < 0):
        raise ValueError('Invalid counts')
    if block.nnz and np.max(np.abs(block.data-np.rint(block.data))) > 0.001:
        raise ValueError('Published raw-data location is not integer counts')
    return block

def normalized(block):
    total = np.asarray(block.sum(axis=1)).ravel()
    return block.multiply((10000/np.maximum(total,1))[:,None]).tocsr(), total

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=Path, default=ROOT)
    ap.add_argument('--genes', type=int, default=2000)
    args = ap.parse_args()
    root = args.root
    out = root/'prepared'; out.mkdir(parents=True, exist_ok=True)
    if (out/'READY.json').exists():
        raise FileExistsError('Prepared version locked; use a new version directory')
    records = json.loads((root/'source_manifest.json').read_text())
    crosswalk = json.loads((root/'label_crosswalk.json').read_text())
    frames = []; common = None; crosswalk_rows = []
    for rec in records:
        p = Path(rec['local_path'])
        if rec['status'] != 'downloaded_structurally_validated' or Path(str(p)+'.aria2').exists():
            raise ValueError('Unvalidated source: '+rec['cancer'])
        with h5py.File(p, 'r') as f:
            obs = read_elem(f['obs']); reader, var, genes = matrix_info(f,rec)
            if obs.index.duplicated().any():
                raise ValueError('Repeated cell identifiers within source')
            common = set(genes) if common is None else common.intersection(genes)
            spec = crosswalk[rec['cancer']]
            original = obs[spec['column']].astype(str)
            missing = set(original)-set(spec['mapping'])
            if missing:
                raise ValueError(f'Unreviewed labels: {missing}')
            label = original.map(spec['mapping'])
            donor = obs['donor_id'].astype(str)
            if donor.isin(['unknown','nan','','na']).any():
                raise ValueError('Unresolved donor identities')
            split = donor.map(donor_split(donor,rec['cancer']))
            frames.append(pd.DataFrame({'source':rec['cancer'], 'source_row':np.arange(len(obs)),
                'cell_id':obs.index.astype(str), 'donor_id':donor.to_numpy(),
                'donor_key':rec['cancer']+'::'+donor.to_numpy(),
                'original_label':original.to_numpy(), 'label':label.to_numpy(),
                'split':split.to_numpy(), 'assay':obs['assay'].astype(str).to_numpy()}))
            for old,new in spec['mapping'].items():
                crosswalk_rows.append({'source':rec['cancer'],'column':spec['column'],
                    'original_label':old,'harmonized_label':new,'cells':int((original==old).sum())})
    common = sorted(g for g in common if g.startswith('ENSG'))
    if len(common) < args.genes:
        raise ValueError('Insufficient common Ensembl genes')
    pd.DataFrame(crosswalk_rows).to_csv(out/'label_crosswalk.csv',index=False)
    cells = pd.concat(frames,ignore_index=True)
    cells['included'] = cells.label.notna()
    cells['exclusion_reason'] = np.where(cells.included,'','ambiguous_author_label')
    cells.to_csv(out/'pre_feature_cell_manifest.csv.gz',index=False)
    cells.groupby(['source','donor_key','split']).size().rename('n_cells').to_csv(out/'donor_split_locked.csv')
    common_file = out/'common_genes.json'
    common_file.write_text(json.dumps(common),encoding='utf-8')
    variance = np.zeros(len(common)); means = np.zeros(len(common))
    offset = 0
    for rec in records:
        frame = cells.iloc[offset:offset+rec['expected_cells']]
        sums = np.zeros(len(common)); sums2 = np.zeros(len(common)); n = 0
        with h5py.File(rec['local_path'],'r') as f:
            reader,var,genes = matrix_info(f,rec); columns = genes.get_indexer(common)
            for start in range(0,len(frame),2048):
                stop = min(start+2048,len(frame))
                block = count_block(reader,start,stop,columns)
                norm,total = normalized(block)
                empty = np.flatnonzero(total==0)+start+offset
                cells.loc[empty,'included'] = False
                cells.loc[empty,'exclusion_reason'] = 'zero_common_gene_counts'
                local = cells.iloc[offset+start:offset+stop]
                train = ((local.split=='train') & local.included).to_numpy()
                z = norm[train]; z.data = np.log1p(z.data)
                sums += np.asarray(z.sum(axis=0)).ravel()
                sums2 += np.asarray(z.power(2).sum(axis=0)).ravel(); n += int(train.sum())
                if start % 40960 == 0: print('TRAIN_FEATURE_STATS',rec['cancer'],start,len(frame),flush=True)
            if not n: raise ValueError('No train cells')
            mean = sums/n
            variance += np.maximum(sums2/n-mean**2,0)/len(records)
            means += mean/len(records)
        offset += len(frame)
    # Equal-source variance ranking prevents the largest cohort determining every feature.
    chosen = np.sort(np.argsort(variance,kind='stable')[-args.genes:])
    selected = np.array(common)[chosen]
    pd.DataFrame({'gene_id':common,'train_equal_source_mean':means,
        'train_equal_source_variance':variance,'selected':np.isin(np.arange(len(common)),chosen)}).to_csv(out/'train_only_feature_statistics.csv',index=False)
    (out/'selected_genes.json').write_text(json.dumps(selected.tolist()),encoding='utf-8')
    shape = (len(cells),len(selected))
    x = np.lib.format.open_memmap(out/'expression_cp10k.npy',mode='w+',dtype='float16',shape=shape)
    fingerprints = []; offset = 0
    for rec in records:
        with h5py.File(rec['local_path'],'r') as f:
            reader,var,genes = matrix_info(f,rec); columns = genes.get_indexer(common)
            for start in range(0,rec['expected_cells'],2048):
                stop = min(start+2048,rec['expected_cells'])
                block = count_block(reader,start,stop,columns)
                subset = block[:,chosen].toarray()
                for row in subset:
                    fingerprints.append(hashlib.sha256(row.astype('<f4').tobytes()).hexdigest())
                norm,total = normalized(block)
                x[offset+start:offset+stop] = norm[:,chosen].toarray().astype(np.float16)
                if start % 40960 == 0: print('MATERIALIZE',rec['cancer'],start,flush=True)
        offset += rec['expected_cells']; x.flush()
    cells['selected_count_fingerprint'] = fingerprints
    # Conservative collisions on the selected raw-count vector, not a claim of identical donors.
    candidates = cells[cells.included & cells.selected_count_fingerprint.duplicated(keep=False)]
    candidates.to_csv(out/'expression_duplicate_candidates.csv.gz',index=False)
    priority = {'train':0,'calibration':1,'test':2}
    ordered = cells[cells.included].assign(priority=lambda d:d.split.map(priority)).sort_values(['priority','source','source_row'])
    remove = ordered.index[ordered.selected_count_fingerprint.duplicated(keep='first')]
    cells.loc[remove,'included'] = False
    cells.loc[remove,'exclusion_reason'] = 'selected_count_duplicate_conservative'
    cells['row_index'] = np.arange(len(cells))
    labels = sorted(cells.loc[cells.included & (cells.split=='train'),'label'].unique())
    cells['label_index'] = cells.label.map({v:i for i,v in enumerate(labels)}).fillna(-1).astype(int)
    cells['eligible'] = cells.included & (cells.label_index>=0)
    if not cells.loc[cells.included,'eligible'].all():
        raise ValueError('Test-only label encountered; require explicit unknown-state protocol')
    donor_roles = cells[cells.eligible].groupby('donor_key').split.nunique()
    assert donor_roles.max() == 1
    assert not cells[cells.eligible].selected_count_fingerprint.duplicated().any()
    cells.to_csv(out/'cells.csv.gz',index=False)
    counts = cells[cells.eligible].groupby(['source','split']).agg(cells=('cell_id','size'),donors=('donor_key','nunique'))
    counts.to_csv(out/'split_counts.csv')
    cells.groupby(['source','exclusion_reason']).size().rename('n').to_csv(out/'exclusions.csv')
    ready = {'seed':SEED,'shape':list(shape),'labels':labels,'n_common_genes':len(common),
        'normalization':'CP10K over common Ensembl genes; model receives log1p(CP10K)',
        'feature_selection':'equal-source mean within-source log1p variance, TRAIN DONORS ONLY',
        'split_policy':'per source hash-seeded donor split, 15% test, 15% calibration, remainder train',
        'source_manifest_sha256':hashlib.sha256((root/'source_manifest.json').read_bytes()).hexdigest(),
        'cell_manifest_sha256':hashlib.sha256((out/'cells.csv.gz').read_bytes()).hexdigest(),
        'crosswalk_sha256':hashlib.sha256((root/'label_crosswalk.json').read_bytes()).hexdigest(),
        'counts':counts.reset_index().to_dict('records'),
        'limitations':['Author cell labels are not exact spatial-proportion truth.',
                      'One source study per cancer: cancer and study are confounded.',
                      'Legacy query data excluded; full donor overlap with legacy studies not established.']}
    (out/'READY.json').write_text(json.dumps(ready,indent=2),encoding='utf-8')
    print(json.dumps(ready,indent=2),flush=True)

if __name__ == '__main__': main()
