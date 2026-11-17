"""Measured largest complete fit releases bounded independent producers and consumers."""
import json
import os
import subprocess
from pathlib import Path
SETUP=Path('/mnt/lustre/work/gehler/gfh527/ni26_calibration339_setup_v1_20261117')
PROD=Path('/mnt/lustre/work/gehler/gfh527/ni26_calibration339_full_v2_20261117')
CHAIN=Path('/mnt/lustre/work/gehler/gfh527/ni26_calibration339_chain_v1_20261117')


def submit(cwd,*args):return subprocess.check_output(['sbatch','--parsable',*args],cwd=cwd,text=True).strip().split(';')[0]


def main():
    m=json.loads((PROD/'P06/seed0/record.json').read_text())
    if m['outer']!='P06' or m['cal']!='P07' or m['training']['steps']!=1480 or m['calibration_labels_opened']:raise ValueError('largestcompleteguard')
    estimate=1.5*m['seconds']*20+124
    if estimate>3600:raise ValueError('oneGPUhourbudgetstop')
    active=subprocess.check_output(['squeue','-u','gfh527','-h','-r','-o','%i'],text=True).splitlines()
    running=subprocess.check_output(['squeue','-u','gfh527','-h','-r','-t','RUNNING','-o','%i'],text=True).splitlines()
    if len(active)+24>244 or len(running)+4>32:raise ValueError('reservedcapacity')
    for i in range(2,11):
        folder=PROD/f'P{i:02d}';folder.mkdir(exist_ok=True);p=folder/'source.sha256'
        if p.exists() and p.read_bytes()!=(PROD/'source.sha256').read_bytes():raise ValueError('foldsourcechanged')
        if not p.exists():p.write_bytes((PROD/'source.sha256').read_bytes())
    array=submit(PROD,'--time=00:05:00','--mem=8G','--array=0-16%4','produce.sbatch')
    assembly=submit(SETUP,f'--dependency=afterok:{array}','assemble.sbatch')
    initial=json.loads((SETUP/'initial_jobs.json').read_text())
    stage=submit(CHAIN,f'--dependency=afterok:{assembly}:{initial["prepare"]}','studies/17nov/representation_chain/stage339.sbatch')
    dep=f'--dependency=afterok:{stage}:{initial["control"]}:{initial["replay"]}'
    smokes=[submit(CHAIN,dep,'studies/17nov/representation_chain/job339.sbatch','cell',a,'within0') for a in ('zero_repr','RF','RV')]
    release=submit(SETUP,'--dependency=afterok:'+':'.join(smokes),'release_chain.sbatch')
    result=dict(array=array,assembly=assembly,stage=stage,smokes=smokes,release=release,largest_seconds=m['seconds'],projected_GPU_seconds=estimate)
    (SETUP/'released_sources.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)


if __name__=='__main__':main()
