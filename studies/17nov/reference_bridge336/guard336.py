"""Native-versus-observer real source control; no query labels or quality score."""
import json
import os
import time
from pathlib import Path
import numpy as np
import torch
from recipe336 import source,native,observe,state_sha,sha,prototype_batches


def main():
    root=Path(os.environ['BRIDGE_ROOT']);torch.set_num_threads(6);start=time.monotonic()
    d,fit,query,people,y=source();records=[];models=[]
    for observed in (False,True):
        model=native(d,epochs=1);trace=observe(model,'A',fit) if observed else None
        tick=time.monotonic();model.fit(fit);seconds=time.monotonic()-tick;model.set_k(20.)
        probe=fit[:12];p=model.predict_proba_stream(probe)
        records.append(dict(observed=observed,state_sha256=state_sha(model.net),scale=model.scale,
            G_sha256=sha_array(model.G),source_proba_sha256=sha_array(p),seconds=seconds,trace=trace))
        models.append(model)
    a,b=records
    for name in ('state_sha256','G_sha256','source_proba_sha256','scale'):
        if a[name]!=b[name]:raise ValueError('nativeobservernotEXACT:'+name)
    alternative=prototype_batches(d,fit,models[1]._order,1)
    actual=records[1]['trace']['batches'][0];candidate=sha_array(np.concatenate([models[1]._order[r] for r in alternative[0]]))
    if candidate==actual:raise ValueError('batchfactorcollapses')
    uncentered=float(torch.as_tensor(d['X'][fit]).float().square().mean().sqrt())
    if abs(uncentered-models[0].scale)<1e-8:raise ValueError('scalefactorcollapses')
    weights=np.array(records[1]['trace']['class_weights'])
    if float(np.max(np.abs(weights-1)))<1e-8:raise ValueError('lossfactorcollapses')
    #Cold1epoch includesfixedinput/setup; conservativefloor uses measured331full40+8order260s.
    warm=records[1]['trace']['full_step_seconds'][2:]
    projected=max(390.,1.5*float(np.median(warm))*records[1]['trace']['steps']*40+120)
    report=dict(passed=True,query_labels_opened=False,source_fit_people=sorted(set(people[fit])),
        source_rows=len(fit),source7_excluded=['P01','P02','P03'],native_observer_exact=True,records=records,
        original_scale=models[0].scale,uncentered_scale=uncentered,prototype_batch_sha256=candidate,
        projected_full_arm_seconds=projected,budget_passed=projected<=480,source_sha256=sha(root/'source.sha256'),seconds=time.monotonic()-start)
    with (root/'guard.json').open('x') as out:json.dump(report,out,indent=2)
    if not report['budget_passed']:
        print(f'GUARDPASS/NORELEASE:fullarmprojection{projected:.1f}s>480',flush=True);return
    import subprocess
    queued=subprocess.check_output(['squeue','-u','gfh527','-h','-r','-o','%i'],text=True).splitlines()
    running=subprocess.check_output(['squeue','-u','gfh527','-h','-r','-t','RUNNING','-o','%i'],text=True).splitlines()
    if len(queued)+6>244 or len(running)+4>32:raise RuntimeError('reservedcapacityunavailable')
    array=subprocess.check_output(['sbatch','--parsable','--array=0-4%4','produce.sbatch'],cwd=root,text=True).strip().split(';')[0]
    collect=subprocess.check_output(['sbatch','--parsable',f'--dependency=afterok:{array}','collect.sbatch'],cwd=root,text=True).strip().split(';')[0]
    (root/'released.json').write_text(json.dumps(dict(array=array,collector=collect)))
    print(f'336nativeobserverEXACT;parallel40epochbridge{array};collector{collect}',flush=True)


def sha_array(a):
    from recipe336 import digest
    return digest(a)


if __name__=='__main__':main()
