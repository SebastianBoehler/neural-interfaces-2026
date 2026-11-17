"""Real source-only composition smoke and conservative production release."""
import json
import os
import subprocess
import time
from pathlib import Path
import numpy as np
import torch
from recipe338 import old,source,observe,CELLS
from produce338 import run


def main():
    root=Path(os.environ['COMBO_ROOT']);torch.set_num_threads(6);start=time.monotonic();records={}
    for arm in ('A','H','W','B','HW','HB','HWB'):
        d,fit,query,people,y,m,t=run('source6',arm,1)
        if set(people[fit])!={'P04','P05','P06','P07','P08','P10'} or np.any(d['y'][query]!=-1):raise ValueError('heldpersonlabelguard')
        records[arm]=t
    a=records['A']
    for arm,t in records.items():
        for k in ('initial_state_sha256','scale','torch_rng_before','torch_rng_after','augment_rng_before','augment_rng_after'):
            if t[k]!=a[k]:raise ValueError('unintendedcompositionchange:'+arm+'/'+k)
        if (t['batches']==a['batches'])!=('B' not in arm):raise ValueError('batchinteractioninactive')
    # Composition adapter's A is exactly the original observer, including parameters and moments.
    d,fit,_,_,_=source('source6');m=old.native(d,1);t=old.observe(m,'A',fit);m.fit(fit)
    if old.state_sha(m.net)!=a['final_state_sha256']:raise ValueError('nativeA6identity')
    projection=max(480.,max(1.5*np.median(t['full_step_seconds'][2:])*t['steps']*40+120 for t in records.values()))
    report=dict(passed=bool(projection<=600),source_sha256=old.sha(root/'source.sha256'),records=records,
                query_labels_opened=False,projection_seconds=float(projection),seconds=time.monotonic()-start)
    (root/'guard.json').write_text(json.dumps(report,indent=2))
    if not report['passed']:raise ValueError('600secondprojectionbudgetstop')
    active=subprocess.check_output(['squeue','-u','gfh527','-h','-r','-o','%i'],text=True).splitlines()
    running=subprocess.check_output(['squeue','-u','gfh527','-h','-r','-t','RUNNING','-o','%i'],text=True).splitlines()
    if len(active)+11>244 or len(running)+4>32:raise ValueError('reservedcapacity')
    array=subprocess.check_output(['sbatch','--parsable','--array=0-9%4','produce.sbatch'],text=True).strip().split(';')[0]
    collector=subprocess.check_output(['sbatch','--parsable',f'--dependency=afterok:{array}','collect.sbatch'],text=True).strip().split(';')[0]
    (root/'released.json').write_text(json.dumps(dict(array=array,collector=collector)));print(array,collector,flush=True)


if __name__=='__main__':main()
