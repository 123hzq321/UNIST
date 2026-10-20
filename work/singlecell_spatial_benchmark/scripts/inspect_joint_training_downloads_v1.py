"""Inspect completed public H5AD assets; never accept an aria2 partial file."""
from __future__ import annotations
import json
from pathlib import Path
import h5py
from anndata.io import read_elem

from joint_paths import joint_root

ROOT = joint_root()

def main():
    for rec in json.loads((ROOT/'source_manifest.json').read_text()):
        p = Path(rec['local_path'])
        if (not p.exists() or p.stat().st_size != rec['expected_bytes']
                or Path(str(p)+'.aria2').exists()):
            print('INCOMPLETE', rec['cancer'], flush=True)
            continue
        out = ROOT/'inspection'/rec['cancer']
        out.mkdir(parents=True, exist_ok=True)
        with h5py.File(p, 'r') as f:
            obs = read_elem(f['obs'])
            prefix = 'raw/' if rec['raw_data_location'] == 'raw.X' else ''
            var = read_elem(f[prefix+'var'])
            x = f[prefix+'X']
            print(rec['cancer'], obs.shape, var.shape, dict(x.attrs), flush=True)
            obs.to_csv(out/'obs.csv.gz')
            var.to_csv(out/'var.csv.gz')
            summary = {}
            for col in obs:
                if obs[col].nunique() <= 200:
                    summary[col] = {str(k):int(v) for k,v in obs[col].value_counts().items()}
            (out/'metadata_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
            print(json.dumps(summary, indent=2), flush=True)

if __name__ == '__main__':
    main()
