"""Synchronous checked pipeline: prepare, joint fit, source controls, report."""
from __future__ import annotations
import subprocess
import sys
from pathlib import Path

from joint_paths import joint_root

ROOT=joint_root()
SCRIPTS=Path(__file__).parent

def execute(name,*args):
    command=[sys.executable,'-u',str(SCRIPTS/name),*args]
    print('EXECUTE',command,flush=True)
    subprocess.run(command,check=True)

def main():
    if not (ROOT/'prepared/READY.json').exists(): execute('prepare_joint_training_v1.py')
    runs=[('joint_mlp_ensemble_v1',None)]+[(f'source_{s}_ensemble_v1',s) for s in ['breast_tnbc','lung_luad','kidney_ccrcc']]
    for run,source in runs:
        if (ROOT/'runs'/run/'COMPLETE.json').exists():
            print('ALREADY_COMPLETE',run,flush=True); continue
        args=['--run-name',run]
        if source: args+=['--train-source',source]
        execute('train_joint_reference_v1.py',*args)
        execute('report_joint_training_v1.py')
    execute('compare_joint_source_controls_v1.py')
    execute('report_joint_training_v1.py')

if __name__=='__main__': main()
