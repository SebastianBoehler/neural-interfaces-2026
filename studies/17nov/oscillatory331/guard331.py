"""One CPU real-source guard; measured bound gates a fixed-recipe pilot."""
import copy
import json
import os
import subprocess
import time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from data331 import load,calibrate,labels,sha,BASE,digest
from train331 import make
from oracle331 import compare
from activity331 import gradients,inference


def reference_cost(d,fit,y):
    import sys
    sys.path.insert(0,str(BASE/'experiments'))
    import session_bn
    if Path(session_bn.__file__).resolve()!=(BASE/'experiments/session_bn.py').resolve():raise ValueError('wrong Rcostmodule')
    data=dict(d);data['y']=np.full(len(d['X']),-1,dtype=np.int64);data['y'][fit]=y
    data.update(w=np.ones(len(d['X'])),nolow=np.zeros(len(d['X']),bool),tag='')
    model=session_bn.make('fbcnet_raw_sbn',data,0)
    if model.dev!='cpu' or model.epochs!=40:raise ValueError('Roriginalrecipe differs')
    model.epochs=1;started=time.monotonic();model.fit(fit)
    return time.monotonic()-started


def main():
    root=Path(os.environ['OSC_ROOT']);torch.set_num_threads(6)
    d,fit,query,people=load();rms,mean,sd,stats=calibrate(d,fit)
    with (root/'scaler.npz').open('xb') as out:np.savez(out,rms=rms,mean=mean.numpy(),sd=sd.numpy(),metadata=json.dumps(stats))
    p04=np.flatnonzero(people=='P04')[:4];x=torch.from_numpy((d['X'][p04]/rms).astype(np.float32))
    y=torch.from_numpy(labels(p04,len(d['X'])))
    oracle={}
    for arm in ('F','S'):
        oracle[arm]={str(dtype):compare(make(arm,mean,sd),x[:1],y[:1],dtype) for dtype in (torch.float64,torch.float32)}
    static,selective=make('F',mean,sd),make('S',mean,sd)
    initial=float((static(x)-selective(x)).abs().max())
    if initial>1e-7:raise ValueError('initialS/Fnotidentical')
    activity={};timing={}
    for arm in ('L','F','S'):
        model=make(arm,mean,sd);activity[arm]=dict(training=gradients(model,x,y,update=True),inference=inference(model,x))
        model=make(arm,mean,sd);optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01)
        r=np.sort(np.random.default_rng(331).permutation(fit)[:32])
        xx=torch.from_numpy((d['X'][r]/rms).astype(np.float32));yy=torch.from_numpy(labels(r,len(d['X'])))
        values=[]
        for iteration in range(7):
            started=time.monotonic();optimizer.zero_grad(set_to_none=True);F.cross_entropy(model(xx),yy).backward();optimizer.step()
            if iteration>=2:values.append(time.monotonic()-started)
        timing[arm]=dict(warm_complete_step_seconds=values,median_seconds=float(np.median(values)))
    yfit=labels(fit,len(d['X']));reference=reference_cost(d,fit,yfit)
    batches=(len(fit)+31)//32
    projection=1.5*sum(v['median_seconds'] for v in timing.values())*40*batches+1.5*reference*40+120
    report=dict(passed=True,query_labels_opened=False,P01_used=False,source=stats,source_rows=len(fit),
        source_sha256=sha(root/'source.sha256'),scaler_sha256=sha(root/'scaler.npz'),oracle=oracle,
        initial_S_F_max_abs=initial,activity=activity,timing=timing,reference_one_epoch_seconds=reference,
        projected_full_pilot_seconds=projection,budget_passed=projection<=2160,torch_version=torch.__version__)
    with (root/'guard.json').open('x') as out:json.dump(report,out,indent=2)
    if projection>2160:
        print(f'NO PILOT: measuredfixedrecipeprojection{projection:.2f}s exceeds2160s; guardpassed',flush=True);return
    queued=subprocess.check_output(['squeue','-u','gfh527','-h','-r','-o','%i'],text=True).splitlines()
    if len(queued)+1>244:raise RuntimeError('capacity unavailable; no pilotreleased')
    job=subprocess.check_output(['sbatch','--parsable',f'--dependency=afterok:{os.environ["SLURM_JOB_ID"]}','pilot.sbatch'],cwd=root,text=True).strip().split(';')[0]
    (root/'continuation.json').write_text(json.dumps(dict(pilot_job=job)))
    print(f'331guard passed; projection{projection:.2f}s; sourcepilot{job}',flush=True)


if __name__=='__main__':main()
