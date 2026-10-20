"""Inventory old reference copies and new public training candidates without changing them."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import re
import h5py
import numpy as np
import pandas as pd
import requests

PROJECT = Path(__file__).resolve().parents[3]
from joint_paths import joint_root

OUT = joint_root()

def strings(node):
    if isinstance(node, h5py.Group) and 'categories' in node:
        cats = strings(node['categories'])
        codes = node['codes'][:]
        return np.array([cats[i] if i >= 0 else '' for i in codes], dtype=object)
    values = node[:]
    return np.array([x.decode() if isinstance(x, bytes) else str(x) for x in values], dtype=object)

def digest(values):
    return hashlib.sha256('\n'.join(sorted(map(str, values))).encode()).hexdigest()

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    audit = OUT / 'audit'
    audit.mkdir(exist_ok=True)
    roots = [PROJECT/'work/singlecell_spatial_benchmark/materialized',
             PROJECT/'work/singlecell_spatial_benchmark/real_world_materialized',
             Path('D:/UNIST_data/real_independent_cohorts/materialized'),
             Path('D:/UNIST_data/cross_cancer_xenium/materialized'),
             Path('D:/UNIST_data/spotless_grouped_cv_v1')]
    paths = sorted({p.resolve() for root in roots if root.exists() for p in root.rglob('sc_reference.h5ad')})
    rows = []
    for i,p in enumerate(paths):
        row = {'path':str(p), 'dataset_id':p.parent.name, 'bytes':p.stat().st_size}
        try:
            with h5py.File(p,'r') as f:
                obs, var = f['obs'], f['var']
                oid = strings(obs[obs.attrs.get('_index','_index')])
                vid = strings(var[var.attrs.get('_index','_index')])
                row.update(n_cells=len(oid),n_genes=len(vid),cell_id_hash=digest(oid),gene_hash=digest(vid))
                row['obs_columns'] = '|'.join(obs.keys())
                for key in ['cell_type','celltypes','celltype_new','cell_type_ontology_term_id','patient_id','patient','Patient','donor_id','orig.ident','sample_id']:
                    if key in obs:
                        val = strings(obs[key]); unique = np.unique(val)
                        row[key+'_n_unique'] = len(unique)
                        row[key+'_values'] = '|'.join(unique)[:12000]
                row['status'] = 'inspected'
        except Exception as exc:
            row.update(status='unreadable',error=str(exc))
        rows.append(row)
        if i % 25 == 0: print('AUDIT',i,len(paths),flush=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(audit/'legacy_reference_inventory.csv',index=False)
    if 'cell_id_hash' in frame:
        frame.groupby(['cell_id_hash'],dropna=False).agg(copies=('path','size'),cells=('n_cells','first'),datasets=('dataset_id',lambda s:'|'.join(s))).to_csv(audit/'legacy_cell_id_duplicate_candidates.csv')
    # Metadata fingerprints identify candidates, not proof of identical assays or biological donors.
    manifests = PROJECT/'outputs/singlecell_spatial_benchmark'
    protected = []
    for pat in ['hest*materialized*manifest*.csv','real_independent_materialized_manifest*.csv','cross_cancer*manifest*.csv','*pathology*manifest*.csv']:
        for p in manifests.glob(pat):
            protected.append({'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'policy':'legacy_external_do_not_train'})
    pd.DataFrame(protected).drop_duplicates('path').to_csv(audit/'protected_legacy_test_manifests.csv',index=False)
    url='https://api.cellxgene.cziscience.com/curation/v1/collections'
    response=requests.get(url,params={'visibility':'PUBLIC'},timeout=120)
    response.raise_for_status(); collections=response.json()
    (audit/'cellxgene_public_collections.json').write_text(json.dumps(collections,indent=2),encoding='utf-8')
    candidates=[]
    for c in collections:
        if re.search('cancer|carcinoma|tumou?r|melanoma|glioblastoma|neoplas',c.get('name',''),re.I):
            for d in c.get('datasets',[]):
                if not any(o['label']=='Homo sapiens' for o in d.get('organism',[])): continue
                candidates.append({'collection_id':c['collection_id'],'collection_name':c.get('name'),
                    'dataset_id':d['dataset_id'],'dataset_version_id':d.get('dataset_version_id'),
                    'disease':'|'.join(x['label'] for x in d.get('disease',[])),
                    'assay':'|'.join(x['label'] for x in d.get('assay',[])),
                    'tissue':'|'.join(x['label'] for x in d.get('tissue',[])),
                    'links':json.dumps(c.get('links',[]))})
    pd.DataFrame(candidates).to_csv(audit/'new_public_reference_candidates.csv',index=False)
    print(json.dumps({'references':len(rows),'new_dataset_candidates':len(candidates),'output':str(audit)},indent=2),flush=True)

if __name__=='__main__': main()
