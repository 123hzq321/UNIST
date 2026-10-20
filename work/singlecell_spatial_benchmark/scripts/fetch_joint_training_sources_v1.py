from __future__ import annotations
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import time
import requests
import h5py

from joint_paths import joint_root

ROOT=joint_root()
SELECTED=[
 ('breast_tnbc','ceef2841-5333-46ac-92ef-ccbe0c20fe55','6f9de485-58cd-4342-bfc4-b3d3dd223aa8'),
 ('lung_luad','0bebef1a-4607-4584-9070-dacf89a0d635','01ff5cf0-730f-4ddc-b1be-7b407211f544'),
 ('kidney_ccrcc','1df8c90d-d299-4b2e-a54d-a5a80f36e780','be39785b-67cb-4177-be19-a40ee3747e45'),
]

def main():
    raw=ROOT/'raw'; raw.mkdir(parents=True,exist_ok=True)
    records=[]
    for cancer,collection_id,dataset_id in SELECTED:
        r=requests.get('https://api.cellxgene.cziscience.com/curation/v1/collections/'+collection_id,timeout=90)
        r.raise_for_status(); collection=r.json()
        (raw/(cancer+'_collection.json')).write_text(json.dumps(collection,indent=2),encoding='utf-8')
        dataset=next(d for d in collection['datasets'] if d['dataset_id']==dataset_id)
        asset=next(a for a in dataset['assets'] if a['filetype']=='H5AD')
        records.append({'cancer':cancer,'collection_id':collection_id,'dataset_id':dataset_id,
            'version_id':dataset['dataset_version_id'],'url':asset['url'],'expected_bytes':asset['filesize'],
            'expected_cells':dataset['cell_count'],'declared_donors':dataset['donor_id'],
            'raw_data_location':dataset['raw_data_location'],'doi':collection.get('doi'),
            'local_path':str(raw/(cancer+'.h5ad')),'status':'queued'})
    manifest=ROOT/'source_manifest.json'
    manifest.write_text(json.dumps(records,indent=2),encoding='utf-8')
    inputs=[]
    for rec in records:
        p=Path(rec['local_path'])
        if p.exists() and p.stat().st_size==rec['expected_bytes'] and not Path(str(p)+'.aria2').exists():
            continue
        inputs.extend([rec['url'],'  dir='+str(raw),'  out='+p.name])
    if inputs:
        input_file=ROOT/'download_input.txt'
        input_file.write_text('\n'.join(inputs)+'\n',encoding='utf-8')
        aria=shutil.which('aria2c')
        if not aria: raise FileNotFoundError('aria2c')
        command=[aria,'-i',str(input_file),'-j','3','-x','6','-s','6','-k','8M',
                 '--continue=true','--file-allocation=none','--auto-file-renaming=false',
                 '--max-tries=4','--retry-wait=5','--connect-timeout=30','--timeout=90',
                 '--summary-interval=20','--console-log-level=warn','--download-result=full']
        print('DOWNLOAD',sum(r['expected_bytes'] for r in records),'bytes',flush=True)
        result=subprocess.run(command,check=False)
        if result.returncode: raise RuntimeError(f'aria2 failed {result.returncode}; files not accepted')
    for rec in records:
        p=Path(rec['local_path'])
        assert p.stat().st_size==rec['expected_bytes']
        assert not Path(str(p)+'.aria2').exists()
        digest=hashlib.sha256()
        with p.open('rb') as f:
            while chunk:=f.read(16*1024*1024): digest.update(chunk)
        with h5py.File(p,'r') as f:
            obs=f['obs']; n=len(obs[obs.attrs['_index']])
            assert n==rec['expected_cells'],(rec['cancer'],n)
            rec['obs_columns']=list(obs)
        rec.update(status='downloaded_structurally_validated',sha256=digest.hexdigest())
        print('VALIDATED',rec['cancer'],n,flush=True)
    manifest.write_text(json.dumps(records,indent=2),encoding='utf-8')

if __name__=='__main__':main()
