"""Secondary control matching optimizer update counts, without test-based tuning."""
from __future__ import annotations
import json
import math
from pathlib import Path
import subprocess
import sys
import pandas as pd

from joint_paths import joint_root

ROOT=joint_root()
SCRIPTS=Path(__file__).parent

def main():
    config=json.loads((ROOT/'runs/joint_mlp_ensemble_v1/training_config.json').read_text())
    counts=pd.read_csv(ROOT/'prepared/split_counts.csv')
    train=counts[counts.split=='train']; batch=config['batch_size']
    target=math.ceil(train.cells.sum()/batch)*config['epochs']; records=[]
    for row in train.itertuples():
        epochs=math.ceil(target/math.ceil(row.cells/batch))
        name=f'steps_matched_{row.source}_ensemble_v1'
        records.append({'source':row.source,'training_cells':row.cells,'epochs':epochs,
            'optimizer_steps_per_member':math.ceil(row.cells/batch)*epochs,'joint_target_steps':target})
        if (ROOT/'runs'/name/'COMPLETE.json').exists(): continue
        print('STEP_MATCHED_CONTROL',records[-1],flush=True)
        logs=ROOT/'logs'; logs.mkdir(exist_ok=True)
        with (logs/(name+'.log')).open('w',encoding='utf-8') as log:
            subprocess.run([sys.executable,'-u',str(SCRIPTS/'train_joint_reference_v1.py'),
                '--run-name',name,'--train-source',row.source,'--epochs',str(epochs)],
                stdout=log,stderr=subprocess.STDOUT,check=True)
        print('COMPLETED',name,flush=True)
    pd.DataFrame(records).to_csv(ROOT/'runs/joint_mlp_ensemble_v1/compute_control_budgets.csv',index=False)
    subprocess.run([sys.executable,'-u',str(SCRIPTS/'compare_joint_source_controls_v1.py'),
        '--control-prefix','steps_matched_','--output-tag','joint_vs_step_matched_source'],check=True)

if __name__=='__main__': main()
