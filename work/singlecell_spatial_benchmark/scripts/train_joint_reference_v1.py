"""Joint cell/within-donor synthetic-mixture training with locked donor tests.

This is reference/mapping pretraining, not evidence of spatial pathology rescue.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score, roc_auc_score
from scipy.stats import spearmanr

PROJECT = Path(__file__).resolve().parents[3]
sys.path.insert(0,str(PROJECT/'src'))
from unist.models import MLPBackbone

from joint_paths import joint_root

ROOT = joint_root()

def balanced_weights(frame):
    sizes = frame.groupby('donor_key').size()
    donors = frame.groupby('source').donor_key.nunique()
    weights = np.array([1/(sizes[d]*donors[s]) for d,s in zip(frame.donor_key,frame.source)])
    return weights/weights.mean()

def weighted_quantile(values, weights, q):
    order = np.argsort(values)
    cumulative = np.cumsum(weights[order]); cumulative /= cumulative[-1]
    return float(np.interp(q,cumulative,values[order]))

def make_mixtures(cells, split, n_per_donor, seed):
    ix=[]; weights=[]; meta=[]
    for (source,donor), frame in cells[cells.eligible & (cells.split==split)].groupby(['source','donor_key'],sort=True):
        donor_seed=int(hashlib.sha256((str(seed)+str(donor)).encode()).hexdigest()[:8],16)
        rng=np.random.default_rng(donor_seed)
        rows=frame.row_index.to_numpy()
        for j in range(n_per_donor):
            size=int(rng.integers(2,9))
            selected=rng.choice(rows,size=size,replace=(len(rows)<size))
            padded=np.repeat(selected[0],8); padded[:size]=selected
            w=np.zeros(8,dtype=np.float32); w[:size]=1/size
            ix.append(padded); weights.append(w)
            meta.append({'source':source,'donor_key':donor,'split':split,'mixture_id':f'{donor}::{j}','n_components':size})
    return np.array(ix,dtype=np.int64),np.array(weights,dtype=np.float32),pd.DataFrame(meta)

def mixture_batch(x, y, indices, weights, classes):
    xb=(x[indices]*weights[:,:,None]).sum(dim=1)
    target=(F.one_hot(y[indices],classes).float()*weights[:,:,None]).sum(dim=1)
    return xb,target

def score_components(probability, disagreement, novelty, scales=None, weights=None):
    k=probability.shape[1]
    entropy=-(probability*np.log(np.maximum(probability,1e-12))).sum(axis=1)/np.log(k)
    if scales is None:
        assert weights is not None
        scales={key:[weighted_quantile(v,weights,.05),weighted_quantile(v,weights,.95)]
                for key,v in [('novelty',novelty),('disagreement',disagreement)]}
    def rescale(v,key):
        low,high=scales[key]
        return np.clip((v-low)/max(high-low,1e-12),0,1)
    n=rescale(novelty,'novelty'); d=rescale(disagreement,'disagreement')
    return {'entropy_only':entropy,'novelty_only':n,'disagreement_only':d,
            'fixed_fusion':.5*entropy+.3*n+.2*d},scales

@torch.inference_mode()
def predict(models,x,centroids,batch=4096):
    probabilities=[]; disagreements=[]; novelties=[]; single=[]
    for start in range(0,len(x),batch):
        z=torch.log1p(x[start:start+batch])
        p=torch.stack([m(z).softmax(dim=1) for m in models])
        probabilities.append(p.mean(dim=0).cpu().numpy())
        disagreements.append(p.var(dim=0,unbiased=False).mean(dim=1).cpu().numpy())
        novelties.append((torch.cdist(z,centroids).square().min(dim=1).values/z.shape[1]).cpu().numpy())
        single.append(p[0].cpu().numpy())
    return tuple(np.concatenate(v) for v in [probabilities,disagreements,novelties,single])

def evaluate_endpoint(name,models,x,y,cells,mixes,centroids,out,classes):
    results={}
    for split in ['calibration','test']:
        if name=='author_cell_labels':
            meta=cells[cells.eligible & (cells.split==split)].copy().reset_index(drop=True)
            ids=torch.as_tensor(meta.row_index.to_numpy(),device=x.device)
            xx=x[ids]
            truth=F.one_hot(y[ids],classes).float().cpu().numpy()
        else:
            indices,ww,meta=mixes[split]
            chunks=[]; targets=[]
            for start in range(0,len(meta),1024):
                ids=torch.as_tensor(indices[start:start+1024],device=x.device)
                w=torch.as_tensor(ww[start:start+1024],device=x.device)
                xb,yb=mixture_batch(x,y,ids,w,classes)
                chunks.append(xb); targets.append(yb.cpu().numpy())
            xx=torch.cat(chunks); truth=np.concatenate(targets)
        p,d,n,single=predict(models,xx,centroids)
        del xx
        weights=balanced_weights(meta)
        scores,scales=score_components(p,d,n,None if split=='calibration' else results['calibration']['scales'],weights)
        scores['single_mlp_entropy']=-(single*np.log(np.maximum(single,1e-12))).sum(axis=1)/np.log(classes)
        tv=.5*np.abs(p-truth).sum(axis=1)
        err=(p.argmax(axis=1)!=truth.argmax(axis=1)).astype(float) if name=='author_cell_labels' else tv
        single_error=(single.argmax(axis=1)!=truth.argmax(axis=1)).astype(float) if name=='author_cell_labels' else .5*np.abs(single-truth).sum(axis=1)
        if split=='calibration':
            thresholds={key:{str(q):weighted_quantile(v,weights,q) for q in [.7,.8,.9]} for key,v in scores.items()}
        else:
            thresholds=results['calibration']['thresholds']
        frame=meta[['source','donor_key']].copy()
        frame['unit_id']=meta.cell_id.to_numpy() if name=='author_cell_labels' else meta.mixture_id.to_numpy()
        frame['error']=err; frame['total_variation']=tv; frame['single_mlp_error']=single_error
        frame['truth_argmax']=truth.argmax(axis=1); frame['prediction_argmax']=p.argmax(axis=1)
        for key,v in scores.items(): frame[key]=v
        frame.to_csv(out/f'{name}_{split}_predictions.csv.gz',index=False)
        np.savez_compressed(out/f'{name}_{split}_probabilities.npz',probabilities=p,truth=truth,single_probabilities=single)
        results[split]={'frame':frame,'scales':scales,'thresholds':thresholds}
    (out/f'{name}_frozen_calibration.json').write_text(json.dumps({
        'scale_fit':'calibration donors only, equal-source/equal-donor weights',
        'score':'0.5 normalized entropy + 0.3 rescaled novelty + 0.2 rescaled ensemble variance',
        'interpretation':'risk index, not a calibrated probability or OOD guarantee',
        'scales':results['calibration']['scales'],'thresholds':results['calibration']['thresholds']},indent=2),encoding='utf-8')
    rows=[]
    for (source,donor), f in results['test']['frame'].groupby(['source','donor_key']):
        for score in ['entropy_only','novelty_only','disagreement_only','fixed_fusion','single_mlp_entropy']:
            error=f.single_mlp_error.to_numpy() if score=='single_mlp_entropy' else f.error.to_numpy()
            values=f[score].to_numpy()
            binary=error.astype(int) if name=='author_cell_labels' else (error>.25).astype(int)
            auc=float(roc_auc_score(binary,values)) if len(np.unique(binary))==2 else np.nan
            rho=float(spearmanr(error,values).statistic) if np.std(error)>0 and np.std(values)>0 else np.nan
            for q in [1.,.9,.8,.7]:
                threshold=np.inf if q==1 else results['calibration']['thresholds'][score][str(q)]
                keep=values<=threshold
                rows.append({'endpoint':name,'source':source,'donor_key':donor,'score':score,
                    'calibration_target_coverage':q,'n':len(f),'retained':int(keep.sum()),
                    'test_coverage':float(keep.mean()),'risk':float(error[keep].mean()) if keep.any() else np.nan,
                    'unfiltered_risk':float(error.mean()),'error_detection_auroc':auc,'error_spearman':rho,
                    'ensemble_macro_f1':float(f1_score(f.truth_argmax,f.prediction_argmax,average='macro')) if name=='author_cell_labels' else np.nan})
    return pd.DataFrame(rows)

def summarize(metrics,out,seed):
    rng=np.random.default_rng(seed); rows=[]
    for keys,g in metrics.groupby(['endpoint','score','calibration_target_coverage']):
        endpoint,score,q=keys
        for field in ['risk','test_coverage','error_detection_auroc','error_spearman']:
            valid=g.dropna(subset=[field]); values=valid[field].to_numpy()
            # Resample donors within each source; keep equal-source weighting.
            groups=[v[field].to_numpy() for _,v in valid.groupby('source')]
            estimates=np.array([np.mean([rng.choice(a,len(a),replace=True).mean() for a in groups]) for _ in range(1000)]) if groups else np.array([np.nan])
            point=float(np.mean([a.mean() for a in groups])) if groups else np.nan
            low,high=np.nanquantile(estimates,[.025,.975]) if len(groups) else (np.nan,np.nan)
            rows.append({'endpoint':endpoint,'score':score,'calibration_target_coverage':q,
                'metric':field,'equal_source_donor_mean':point,'ci_low':low,'ci_high':high,'n_donors':len(values),
                'ci_method':'1000 donor bootstraps within source; sources fixed'})
    pd.DataFrame(rows).to_csv(out/'test_summary_donor_bootstrap.csv',index=False)
    comparisons=[]
    for (endpoint,q),g in metrics.groupby(['endpoint','calibration_target_coverage']):
        pivot=g.pivot(index=['source','donor_key'],columns='score',values='risk')
        for baseline in ['entropy_only','novelty_only','disagreement_only']:
            delta=(pivot['fixed_fusion']-pivot[baseline]).dropna()
            groups=[v.to_numpy() for _,v in delta.groupby(level='source')]
            b=np.array([np.mean([rng.choice(a,len(a),replace=True).mean() for a in groups]) for _ in range(1000)])
            comparisons.append({'endpoint':endpoint,'calibration_target_coverage':q,'comparison':'fusion_minus_'+baseline,
                'risk_difference':float(np.mean([a.mean() for a in groups])),
                'ci_low':float(np.quantile(b,.025)),'ci_high':float(np.quantile(b,.975)),
                'n_donors':len(delta),'note':'Calibration coverage matched, realized test coverage can differ; negative favors fusion.'})
    pd.DataFrame(comparisons).to_csv(out/'paired_donor_risk_differences.csv',index=False)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--root',type=Path,default=ROOT)
    ap.add_argument('--epochs',type=int,default=30)
    ap.add_argument('--members',type=int,default=3)
    ap.add_argument('--batch-size',type=int,default=2048)
    ap.add_argument('--seed',type=int,default=20260926)
    ap.add_argument('--run-name',default='joint_mlp_ensemble_v1')
    ap.add_argument('--device',choices=['cuda','cpu'],default='cuda',help='The archived full study used CUDA; CPU supports small verification runs.')
    ap.add_argument('--train-source',default=None,help='Optional source-specific neural control under the same locked feature/split space.')
    args=ap.parse_args()
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    if args.device=='cuda' and not torch.cuda.is_available(): raise RuntimeError('CUDA is unavailable; use --device cpu for small verification runs')
    device=args.device; root=args.root; prep=root/'prepared'; out=root/'runs'/args.run_name
    out.mkdir(parents=True,exist_ok=True)
    if (out/'COMPLETE.json').exists(): raise FileExistsError('Completed run is locked')
    ready=json.loads((prep/'READY.json').read_text())
    assert hashlib.sha256((prep/'cells.csv.gz').read_bytes()).hexdigest()==ready['cell_manifest_sha256']
    cells=pd.read_csv(prep/'cells.csv.gz'); raw=np.load(prep/'expression_cp10k.npy',mmap_mode='r')
    if args.train_source:
        if args.train_source not in set(cells.source): raise ValueError('Unknown training source')
        cells['eligible'] &= cells.source==args.train_source
    classes=len(ready['labels']); n_genes=raw.shape[1]
    config={'epochs':args.epochs,'ensemble_members':args.members,'batch_size':args.batch_size,
        'learning_rate':.001,'seed':args.seed,'device':torch.cuda.get_device_name(0) if device=='cuda' else 'cpu',
        'torch':torch.__version__,'backbone':'UNIST existing MLP 128/64 LayerNorm GELU dropout0.2',
        'python':sys.version,'runtime_versions':{p:importlib.metadata.version(p) for p in ['numpy','pandas','scipy','scikit-learn','anndata','h5py']},
        'objective':'0.5 source/donor-weighted cell CE + 0.5 source/donor-balanced synthetic-mixture soft CE',
        'cell_exposure':'every eligible train cell exactly once per epoch per member; no cell downsampling',
        'mixtures':'512 fixed mixtures per training donor; 256 per calibration/test donor; 2-8 cells, within donor, equal normalized-expression weights',
        'early_stopping':'none, fixed epochs; test used only after all models frozen',
        'training_source_filter':args.train_source,
        'source_control_features':'All runs share the joint TRAIN-only feature space; no held-out donor expression used for feature selection.',
        'ready':ready,'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    config_file=out/'training_config.json'
    if config_file.exists() and json.loads(config_file.read_text())!=config:
        raise ValueError('Resume config mismatch; use a new run name')
    config_file.write_text(json.dumps(config,indent=2),encoding='utf-8')
    print('LOADING_DEVICE',device,raw.shape,flush=True)
    x=torch.empty(raw.shape,dtype=torch.float32,device=device)
    for start in range(0,len(raw),8192):
        x[start:start+8192]=torch.as_tensor(np.array(raw[start:start+8192],dtype=np.float32),device=device)
    y=torch.as_tensor(cells.label_index.to_numpy(),dtype=torch.long,device=device)
    train=cells[cells.eligible & (cells.split=='train')]
    train_ids=torch.as_tensor(train.row_index.to_numpy(),device=device)
    weight=torch.as_tensor(balanced_weights(train),dtype=torch.float32,device=device)
    mixes={}
    for i,split in enumerate(['train','calibration','test']):
        mixes[split]=make_mixtures(cells,split,512 if split=='train' else 256,args.seed+i)
        ix,w,meta=mixes[split]; np.savez_compressed(out/f'{split}_mixture_components.npz',row_indices=ix,weights=w)
        meta.to_csv(out/f'{split}_mixture_manifest.csv.gz',index=False)
    mi,mw,mm=mixes['train']
    mi=torch.as_tensor(mi,device=device); mw=torch.as_tensor(mw,device=device)
    mp=balanced_weights(mm); mp/=mp.sum()
    models=[]
    history=pd.read_csv(out/'training_history.csv').to_dict('records') if (out/'training_history.csv').exists() else []
    start_time=time.monotonic()
    for member in range(args.members):
        seed=args.seed+member*1009; np.random.seed(seed); torch.manual_seed(seed)
        if device=='cuda': torch.cuda.manual_seed_all(seed)
        model=MLPBackbone(n_genes,classes).to(device)
        checkpoint=out/f'ensemble_member_{member}.pt'
        if checkpoint.exists():
            saved=torch.load(checkpoint,map_location=device,weights_only=False)
            model.load_state_dict(saved['state_dict']); model.eval(); models.append(model)
            print('RESUME_COMPLETED_MEMBER',member,flush=True); continue
        rng=np.random.default_rng(seed); opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=1e-4)
        for epoch in range(args.epochs):
            model.train(); order=torch.randperm(len(train_ids),device=device); losses=[]
            for begin in range(0,len(order),args.batch_size):
                order_index=order[begin:begin+args.batch_size]; ids=train_ids[order_index]
                logits=model(torch.log1p(x[ids]))
                cell_loss=(F.cross_entropy(logits,y[ids],reduction='none')*weight[order_index]).mean()
                selected=torch.as_tensor(rng.choice(len(mi),512,replace=True,p=mp),device=device)
                xb,target=mixture_batch(x,y,mi[selected],mw[selected],classes)
                mixture_loss=-(target*F.log_softmax(model(torch.log1p(xb)),dim=1)).sum(dim=1).mean()
                loss=.5*cell_loss+.5*mixture_loss
                opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),5.0); opt.step()
                losses.append(float(loss.detach().cpu()))
            item={'member':member,'epoch':epoch+1,'loss':float(np.mean(losses)),
                  'train_cells_seen':len(train_ids),'elapsed_seconds':round(time.monotonic()-start_time,1)}
            history.append(item); print(json.dumps(item),flush=True)
            pd.DataFrame(history).to_csv(out/'training_history.csv',index=False)
            (out/'status.json').write_text(json.dumps({'phase':'training',**item}),encoding='utf-8')
        model.eval(); torch.save({'state_dict':model.state_dict(),'config':config,'member':member},checkpoint)
        models.append(model)
    print('FROZEN_MODELS_EVALUATING',flush=True)
    (out/'status.json').write_text(json.dumps({'phase':'evaluating','completed_models':len(models)}),encoding='utf-8')
    with torch.inference_mode():
        centroids=[]
        for label in range(classes):
            ids=train_ids[y[train_ids]==label]; total=torch.zeros(n_genes,device=device)
            if not len(ids): continue
            for begin in range(0,len(ids),4096): total+=torch.log1p(x[ids[begin:begin+4096]]).sum(dim=0)
            centroids.append(total/max(len(ids),1))
        centroids=torch.stack(centroids)
    np.save(out/'train_class_centroids_log1p.npy',centroids.cpu().numpy())
    all_metrics=[]
    for endpoint in ['author_cell_labels','synthetic_within_donor_mixtures']:
        print('EVALUATE',endpoint,flush=True)
        all_metrics.append(evaluate_endpoint(endpoint,models,x,y,cells,mixes,centroids,out,classes))
    metrics=pd.concat(all_metrics,ignore_index=True); metrics.to_csv(out/'test_donor_metrics.csv',index=False)
    summarize(metrics,out,args.seed)
    complete={'phase':'completed','training_cells':len(train),'training_donors':train.donor_key.nunique(),
        'training_sources':train.source.nunique(),'epochs_per_member':args.epochs,'completed_models':len(models),
        'elapsed_seconds':round(time.monotonic()-start_time,1),
        'scope':'Reference and synthetic-mixture training; independent spatial/pathology efficacy not tested here.'}
    (out/'COMPLETE.json').write_text(json.dumps(complete,indent=2),encoding='utf-8')
    (out/'status.json').write_text(json.dumps(complete,indent=2),encoding='utf-8')
    print(json.dumps(complete,indent=2),flush=True)

if __name__=='__main__': main()
