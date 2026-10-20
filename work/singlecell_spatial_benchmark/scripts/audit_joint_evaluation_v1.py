"""Post-fit integrity and matched-coverage diagnostics, without refitting models."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
import h5py
from prepare_joint_training_v1 import matrix_info, count_block, normalized
from scipy.stats import spearmanr

from joint_paths import joint_root

ROOT=joint_root()

def main():
    prep=ROOT/'prepared'; run=ROOT/'runs/joint_mlp_ensemble_v1'
    cells=pd.read_csv(prep/'cells.csv.gz',low_memory=False)
    valid=cells[cells.eligible]
    assert valid.groupby('donor_key').split.nunique().max()==1
    assert not valid.selected_count_fingerprint.duplicated().any()
    assert (valid.label_index>=0).all()
    assert len(valid)==563983
    assert len(valid[valid.split=='train'])==367580
    matrix=np.load(prep/'expression_cp10k.npy',mmap_mode='r')
    common=json.loads((prep/'common_genes.json').read_text())
    selected=json.loads((prep/'selected_genes.json').read_text())
    selected_index=pd.Index(common).get_indexer(selected)
    spotchecks=[]; rng=np.random.default_rng(20260926)
    for rec in json.loads((ROOT/'source_manifest.json').read_text()):
        source_cells=cells[cells.source==rec['cancer']]
        positions=rng.choice(len(source_cells),8,replace=False)
        with h5py.File(rec['local_path'],'r') as f:
            reader,var,genes=matrix_info(f,rec); columns=genes.get_indexer(common)
            for pos in positions:
                cell=source_cells.iloc[pos]
                block=count_block(reader,int(cell.source_row),int(cell.source_row)+1,columns)
                norm,total=normalized(block)
                expected=norm[:,selected_index].toarray()[0].astype(np.float16)
                np.testing.assert_array_equal(expected,matrix[int(cell.row_index)])
                spotchecks.append({'source':rec['cancer'],'cell_id':cell.cell_id,'row_index':cell.row_index,'exact_float16_match':True})
    pd.DataFrame(spotchecks).to_csv(run/'raw_to_prepared_spotchecks.csv',index=False)
    artifacts=[]
    for directory in sorted((ROOT/'runs').glob('*ensemble_v1')):
        assert (directory/'COMPLETE.json').exists(),directory
        history=pd.read_csv(directory/'training_history.csv')
        config=json.loads((directory/'training_config.json').read_text())
        assert len(history)==config['epochs']*config['ensemble_members']
        assert history.groupby('member').epoch.max().eq(config['epochs']).all()
        assert np.isfinite(history.loss).all()
        for p in directory.glob('ensemble_member_*.pt'):
            artifacts.append({'path':str(p),'bytes':p.stat().st_size,
                'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
    expected=3*len(list((ROOT/'runs').glob('*ensemble_v1')))
    assert len(artifacts)==expected
    assert len({r['sha256'] for r in artifacts})==expected
    pd.DataFrame(artifacts).to_csv(run/'checkpoint_integrity.csv',index=False)
    rows=[]; events=[]
    for endpoint in ['author_cell_labels','synthetic_within_donor_mixtures']:
        df=pd.read_csv(run/f'{endpoint}_test_predictions.csv.gz')
        for (source,donor),frame in df.groupby(['source','donor_key']):
            binary=frame.error>0 if endpoint=='author_cell_labels' else frame.error>.25
            events.append({'endpoint':endpoint,'source':source,'donor_key':donor,
                'n':len(frame),'positive_error_events':int(binary.sum()),
                'auroc_defined':bool(binary.any() and (~binary).any())})
            seed=int(hashlib.sha256((endpoint+donor+'20260926').encode()).hexdigest()[:8],16)
            tie_break=np.random.default_rng(seed).random(len(frame))
            for score in ['entropy_only','novelty_only','disagreement_only','fixed_fusion','single_mlp_entropy']:
                order=np.lexsort((tie_break,frame[score].to_numpy()))
                error=frame.single_mlp_error.to_numpy() if score=='single_mlp_entropy' else frame.error.to_numpy()
                for coverage in [.7,.8,.9,1.]:
                    retained=max(1,int(np.ceil(coverage*len(frame))))
                    rows.append({'endpoint':endpoint,'source':source,'donor_key':donor,'score':score,
                        'target_test_coverage':coverage,'actual_test_coverage':retained/len(frame),
                        'retained':retained,'risk':float(error[order[:retained]].mean())})
    metrics=pd.DataFrame(rows); metrics.to_csv(run/'matched_test_coverage_donor_metrics.csv',index=False)
    pd.DataFrame(events).to_csv(run/'error_event_counts.csv',index=False)
    rng=np.random.default_rng(20260926); summaries=[]
    for (endpoint,coverage),frame in metrics.groupby(['endpoint','target_test_coverage']):
        pivot=frame.pivot(index=['source','donor_key'],columns='score',values='risk')
        for baseline in ['entropy_only','novelty_only','disagreement_only']:
            delta=pivot.fixed_fusion-pivot[baseline]
            arrays=[g.to_numpy() for _,g in delta.groupby(level='source')]
            values=np.array([np.mean([rng.choice(a,len(a),replace=True).mean() for a in arrays]) for _ in range(1000)])
            summaries.append({'endpoint':endpoint,'matched_test_coverage':coverage,'comparison':'fusion_minus_'+baseline,
                'risk_difference':float(np.mean([a.mean() for a in arrays])),
                'ci_low':float(np.quantile(values,.025)),'ci_high':float(np.quantile(values,.975)),
                'n_donors':len(delta),'scope':'post-fit ranking diagnostic; NOT a deployable threshold or prospective coverage guarantee'})
    summary=pd.DataFrame(summaries); summary.to_csv(run/'matched_test_coverage_paired_differences.csv',index=False)
    predictions=pd.read_csv(run/'synthetic_within_donor_mixtures_test_predictions.csv.gz')
    truth=np.load(run/'synthetic_within_donor_mixtures_test_probabilities.npz')['truth']
    predictions['true_composition_entropy']=-(truth*np.log(np.maximum(truth,1e-12))).sum(axis=1)/np.log(truth.shape[1])
    entropy_rows=[]
    for (source,donor),frame in predictions.groupby(['source','donor_key']):
        entropy_rows.append({'source':source,'donor_key':donor,'n':len(frame),
            'predicted_vs_true_composition_entropy_rho':float(spearmanr(frame.entropy_only,frame.true_composition_entropy).statistic),
            'predicted_entropy_vs_mapping_error_rho':float(spearmanr(frame.entropy_only,frame.error).statistic),
            'scope':'Synthetic mixture diagnostic only; association is not causal or pathology evidence.'})
    pd.DataFrame(entropy_rows).to_csv(run/'mixture_entropy_confound_audit.csv',index=False)
    result={'status':'passed','donor_roles_disjoint':True,'selected_expression_fingerprints_disjoint':True,
        'eligible_cells':len(valid),'checkpoints':len(artifacts),'raw_to_prepared_spotchecks':len(spotchecks),
        'matched_coverage_diagnostic':'Test predictions only; no retraining, score tuning or checkpoint selection.',
        'source_control_budget':'Report equal-epoch and secondary optimizer-step-matched controls separately.'}
    (run/'FINAL_AUDIT.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2),flush=True)
    print(summary[summary.matched_test_coverage==.8].to_string(index=False),flush=True)

if __name__=='__main__': main()
