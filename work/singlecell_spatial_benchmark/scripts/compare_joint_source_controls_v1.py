"""Paired donor comparison under identical test cells and synthetic mixtures."""
from __future__ import annotations
import json
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

from joint_paths import joint_root

ROOT=joint_root()

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--control-prefix',default='source_')
    ap.add_argument('--output-tag',default='joint_vs_source')
    args=ap.parse_args()
    joint=ROOT/'runs/joint_mlp_ensemble_v1'; rows=[]; rng=np.random.default_rng(20260926)
    for source in ['breast_tnbc','lung_luad','kidney_ccrcc']:
        control=ROOT/'runs'/f'{args.control_prefix}{source}_ensemble_v1'
        assert (control/'COMPLETE.json').exists()
        for endpoint in ['author_cell_labels','synthetic_within_donor_mixtures']:
            j=pd.read_csv(joint/f'{endpoint}_test_predictions.csv.gz')
            s=pd.read_csv(control/f'{endpoint}_test_predictions.csv.gz')
            j=j[j.source==source]
            pair=j.merge(s,on=['source','donor_key','unit_id'],suffixes=('_joint','_source'),validate='one_to_one')
            assert len(pair)==len(j)==len(s)
            assert (pair.truth_argmax_joint==pair.truth_argmax_source).all()
            if endpoint=='synthetic_within_donor_mixtures':
                jm=pd.read_csv(joint/'test_mixture_manifest.csv.gz')
                jmi=np.load(joint/'test_mixture_components.npz')
                smi=np.load(control/'test_mixture_components.npz')
                mask=(jm.source==source).to_numpy()
                np.testing.assert_array_equal(jmi['row_indices'][mask],smi['row_indices'])
                np.testing.assert_allclose(jmi['weights'][mask],smi['weights'])
            for donor,g in pair.groupby('donor_key'):
                rows.append({'source':source,'donor_key':donor,'endpoint':endpoint,'n':len(g),
                    'joint_error':g.error_joint.mean(),'source_specific_error':g.error_source.mean(),
                    'joint_minus_source_error':(g.error_joint-g.error_source).mean()})
    frame=pd.DataFrame(rows); frame.to_csv(joint/f'{args.output_tag}_donor_metrics.csv',index=False)
    result=[]
    for endpoint,g in frame.groupby('endpoint'):
        arrays=[v.joint_minus_source_error.to_numpy() for _,v in g.groupby('source')]
        samples=np.array([np.mean([rng.choice(a,len(a),replace=True).mean() for a in arrays]) for _ in range(1000)])
        result.append({'endpoint':endpoint,'n_donors':len(g),
            'joint_minus_source_error':float(np.mean([a.mean() for a in arrays])),
            'ci_low':float(np.quantile(samples,.025)),'ci_high':float(np.quantile(samples,.975)),
            'comparison':'same test units; negative favors joint; equal-source/equal-donor weighting; source-stratified donor bootstrap',
            'limitation':'one renal test donor; shared joint-train feature selection; no independent study generalization inference'})
    pd.DataFrame(result).to_csv(joint/f'{args.output_tag}_summary.csv',index=False)
    print(json.dumps(result,indent=2),flush=True)

if __name__=='__main__': main()
