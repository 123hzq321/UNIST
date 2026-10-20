from __future__ import annotations
import json
from pathlib import Path
import tempfile
import unittest
import subprocess
import sys
import numpy as np
import pandas as pd
import anndata as ad
from scipy import sparse
from prepare_joint_training_v1 import donor_split, normalized
from train_joint_reference_v1 import balanced_weights, make_mixtures, score_components

class TrainingContracts(unittest.TestCase):
    def test_donor_split_is_disjoint_and_repeatable(self):
        donors=[f'p{i}' for i in range(20)]
        a=donor_split(donors,'source')
        self.assertEqual(a,donor_split(donors[::-1],'source'))
        self.assertEqual(list(a.values()).count('test'),3)
        self.assertEqual(list(a.values()).count('calibration'),3)

    def test_equal_source_and_donor_weight(self):
        frame=pd.DataFrame({'source':['a']*7+['b']*3,
                            'donor_key':['a1']*2+['a2']*5+['b1']*3})
        frame['weight']=balanced_weights(frame)
        self.assertAlmostEqual(frame.groupby('source').weight.sum()['a'],5)
        self.assertAlmostEqual(frame.groupby('donor_key').weight.sum()['a1'],2.5)

    def test_mixtures_do_not_cross_donors_or_roles(self):
        frame=pd.DataFrame({'source':['a']*9,'donor_key':['a1']*3+['a2']*3+['a3']*3,
            'row_index':np.arange(9),'split':['train']*6+['test']*3,'eligible':[True]*9})
        idx,w,meta=make_mixtures(frame,'train',20,42)
        self.assertTrue(np.all(idx<6))
        np.testing.assert_allclose(w.sum(axis=1),1)
        for row,donor in zip(idx,meta.donor_key):
            self.assertEqual(set(frame.iloc[row].donor_key),{donor})

    def test_query_does_not_refit_scales(self):
        p=np.array([[.9,.1],[.2,.8],[.6,.4]])
        _,scale=score_components(p,np.array([.1,.2,.3]),np.array([1.,2.,3.]),weights=np.ones(3))
        initial=json.dumps(scale,sort_keys=True)
        score,new=score_components(p,np.array([99.,100.,200.]),np.array([100.,200.,300.]),scale)
        self.assertEqual(initial,json.dumps(new,sort_keys=True))
        np.testing.assert_allclose(score['novelty_only'],1)

    def test_source_controls_have_identical_test_mixtures(self):
        frame=pd.DataFrame({'source':['a']*6+['b']*6,
            'donor_key':['a1']*6+['b1']*6,'row_index':np.arange(12),
            'split':['test']*12,'eligible':[True]*12})
        idx,w,meta=make_mixtures(frame,'test',10,42)
        subset=frame.copy(); subset['eligible'] &= subset.source=='b'
        ix2,w2,meta2=make_mixtures(subset,'test',10,42)
        np.testing.assert_array_equal(idx[meta.source=='b'],ix2)
        np.testing.assert_allclose(w[meta.source=='b'],w2)

    def test_common_gene_normalization(self):
        x=sparse.csr_matrix([[1,3],[0,0],[10,10]],dtype=np.float32)
        z,total=normalized(x)
        np.testing.assert_allclose(z.sum(axis=1).A.ravel(),[10000,0,10000])

def smoke():
    root=Path(tempfile.mkdtemp(prefix='joint_training_smoke_'))
    rng=np.random.default_rng(421); records=[]; crosswalk={}
    for source in ['a','b','c']:
        n=360; g=128; labels=np.tile(np.arange(3),120)
        counts=rng.poisson(.2,(n,g)).astype(np.float32)
        for label in range(3): counts[labels==label,label*20:(label+1)*20]+=rng.poisson(5,(int((labels==label).sum()),20))
        obs=pd.DataFrame({'donor_id':np.repeat([f'p{i}' for i in range(6)],60),
            'author_label':labels.astype(str),'assay':['test']*n},index=[f'{source}_{i}' for i in range(n)])
        var=pd.DataFrame(index=[f'ENSG{i:08}' for i in range(g)])
        a=ad.AnnData(sparse.csr_matrix(counts),obs=obs,var=var)
        p=root/f'{source}.h5ad'; a.write_h5ad(p)
        records.append({'cancer':source,'local_path':str(p),'raw_data_location':'X',
            'expected_cells':n,'status':'downloaded_structurally_validated'})
        crosswalk[source]={'column':'author_label','mapping':{str(i):f'type{i}' for i in range(3)}}
    (root/'source_manifest.json').write_text(json.dumps(records))
    (root/'label_crosswalk.json').write_text(json.dumps(crosswalk))
    scripts=Path(__file__).parent
    subprocess.run([sys.executable,str(scripts/'prepare_joint_training_v1.py'),'--root',str(root),'--genes','64'],check=True)
    subprocess.run([sys.executable,str(scripts/'train_joint_reference_v1.py'),'--root',str(root),
        '--epochs','2','--members','2','--run-name','smoke','--device','cpu'],check=True)
    ready=json.loads((root/'prepared/READY.json').read_text())
    cells=pd.read_csv(root/'prepared/cells.csv.gz')
    assert cells.groupby('donor_key').split.nunique().max()==1
    assert (root/'runs/smoke/COMPLETE.json').exists()
    # Perturb ONLY held-out expression: train-only HVG statistics must stay unchanged.
    stats=pd.read_csv(root/'prepared/train_only_feature_statistics.csv')
    perturb=root/'perturbed_holdout'; perturb.mkdir()
    changed=[]
    for rec in records:
        a=ad.read_h5ad(rec['local_path'])
        roles=donor_split(a.obs.donor_id,rec['cancer'])
        mask=a.obs.donor_id.map(roles).to_numpy()!='train'
        matrix=a.X.toarray(); matrix[mask]+=7
        a.X=sparse.csr_matrix(matrix)
        p=perturb/(rec['cancer']+'.h5ad'); a.write_h5ad(p)
        changed.append({**rec,'local_path':str(p)})
    (perturb/'source_manifest.json').write_text(json.dumps(changed))
    (perturb/'label_crosswalk.json').write_text(json.dumps(crosswalk))
    subprocess.run([sys.executable,str(scripts/'prepare_joint_training_v1.py'),'--root',str(perturb),'--genes','64'],check=True)
    other=pd.read_csv(perturb/'prepared/train_only_feature_statistics.csv')
    pd.testing.assert_frame_equal(stats,other)
    print('HELDOUT_PERTURBATION_FEATURE_INVARIANCE_OK',flush=True)
    print('SMOKE_OK',str(root),'counts',ready['counts'],flush=True)
    return root

if __name__=='__main__':
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(TrainingContracts))
    if not result.wasSuccessful(): raise SystemExit(1)
    if '--smoke' in sys.argv: smoke()
