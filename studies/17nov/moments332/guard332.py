"""Real CPU activation oracle and no-label full-model path checks before GPUrelease."""
import copy
import json
import os
import subprocess
import time
from pathlib import Path
import numpy as np
import torch
import source332 as src
from views332 import MomentBN,direct,prepare
from predict332 import run


def main():
    root=Path(os.environ['MOMENT_ROOT']);torch.set_num_threads(6);started=time.monotonic()
    d,fit,test,donor,perms,nets,original,meta=src.load('P10',0,'cpu')
    # Small complete real session prefix for CPU engineering only; GPUsmoke covers fullP10.
    session=d['stream'][test[0]];rows=test[d['stream'][test]==session][:12]
    results={}
    for label,(model,scale) in nets.items():
        x=torch.from_numpy(d['X'][rows]/scale);views,_=prepare(model,x);captured=[]
        layer=next(m for m in views['B'].body.modules() if isinstance(m,src.session_bn.SessionBN))
        hook=layer.register_forward_pre_hook(lambda m,args:captured.append(args[0].detach().clone()))
        src.modes.probabilities(views['B'],d['X'][rows]/scale,'cpu');hook.remove();activation=captured[0]
        errors={}
        for mode in ('M','V','B','D'):
            custom=copy.deepcopy(layer);custom.__class__=MomentBN;custom.mode332=mode
            value=activation if mode!='D' else torch.cat([activation,activation.flip(0)])
            for dtype in (torch.float64,torch.float32):
                adapted=copy.deepcopy(custom).to(dtype);z=value.to(dtype)
                error=float((adapted(z)-direct(adapted,z,mode)).abs().max())
                bound=2e-8 if dtype==torch.float64 else 2e-4
                if dtype==torch.float64 and error>bound:raise ValueError(f'float64momentoracle:{label}/{mode}:{error}')
                errors[f'{mode}|{dtype}']=error
            if mode=='B':
                error=float((custom(activation)-layer(activation)).abs().max())
                if error>2e-5:raise ValueError('bothmathnotoriginalSessionBN')
                errors['B_original']=error
        # Float64moment arithmetic is an independent precision reference; all other kernels stayfloat32.
        class ReferenceMomentBN(MomentBN):
            def forward(self,x):return super().forward(x.double()).to(x.dtype)
        for mode in ('M','V','B','D'):
            current=views[mode] if mode!='B' else views['B']
            reference=copy.deepcopy(current)
            for module in reference.body.modules():
                if isinstance(module,src.session_bn.SessionBN):
                    module.__class__=ReferenceMomentBN;module.mode332=mode;module.double()
            target=d['X'][rows]/scale
            if mode=='D':target=np.concatenate([target,d['X'][np.resize(donor,len(rows))]/scale])
            actual=src.modes.probabilities(current,target,'cpu')
            truth=src.modes.probabilities(reference,target,'cpu')
            error=float(np.max(np.abs(actual-truth)));errors[mode+'_probability_vs_float64_moments']=error
            if error>2e-5 or not np.array_equal(actual.argmax(1),truth.argmax(1)):
                raise ValueError(f'momentprecisionchangesprediction:{label}/{mode}:{error}')
        results[label]=errors
    # Fulltargetrows retains canonicalindexing; onlywithin0 CPU outputengineering, no oldGPUidentityclaimyet.
    banks,checks,donors=run(d,test,donor,perms,nets,original,'cpu',check_original=False,orders=('within0',))
    report=dict(passed=True,labels_opened=False,held='P10',seed=0,source=meta,
        moment_oracle=results,checks=checks,donor_order_rows=donors,seconds=time.monotonic()-started,
        source_sha256=src.sha(root/'source.sha256'))
    with (root/'cpu_guard.json').open('x') as out:json.dump(report,out,indent=2)
    active=subprocess.check_output(['squeue','-u','gfh527','-h','-r','-o','%i'],text=True).splitlines()
    if len(active)+1>244:raise RuntimeError('28reservedcapacityunavailable')
    job=subprocess.check_output(['sbatch','--parsable',f'--dependency=afterok:{os.environ["SLURM_JOB_ID"]}','smoke.sbatch'],cwd=root,text=True).strip().split(';')[0]
    (root/'smoke_job.json').write_text(json.dumps(dict(job=job)))
    print(f'332CPUrealdata/oraclepassed; GPUfullP10smoke{job}',flush=True)


if __name__=='__main__':main()
