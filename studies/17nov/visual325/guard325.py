"""CPU source-only no-label proof, then exactly one bounded A100 continuation."""
import json
import os
import subprocess
from pathlib import Path
import numpy as np
import torch
from support325 import setup, build, digest, sha, features
from renderer325 import Renderer, neighbors
from checks325 import run


def main():
    root=Path(os.environ['VISUAL_ROOT']);torch.set_num_threads(6)
    d,fit,val,people,base,meta=setup(root);near,montage=neighbors(d['ch_names'])
    rows=np.array([np.flatnonzero(people==p)[0] for p in ('P04','P05','P06','P07')])
    x=torch.from_numpy(d['X'][rows].copy());renderer=Renderer(base,near)
    with torch.no_grad():
        error=float((renderer(x)-base(x,41,'intensity')).abs().max())
        if error>1e-5: raise ValueError('zero learned renderer differs from fixeddisplay')
    checks={};proof={}
    for variant in ('pretrained','random'):
        encoder=build(root,variant,'cpu')
        checks[variant]=run(renderer,encoder,x,'cpu')
        with torch.no_grad():proof[variant]=features(encoder,renderer(x[:1]),1435).numpy()
        del encoder
    with (root/'guard_features.npz').open('xb') as out:np.savez(out,**proof)
    report=dict(passed=True,labels_opened=False,source_rows=rows.tolist(),source=meta,
        montage=montage,neighbors=near.tolist(),zero_identity_max_abs=error,checks=checks,
        source_sha256=sha(root/'source.sha256'),feature_sha256=sha(root/'guard_features.npz'))
    (root/'guard.json').write_text(json.dumps(report,indent=2))
    active=subprocess.check_output(['squeue','-u','gfh527','-h','-r','-o','%i'],text=True).splitlines()
    if len(active)+1>244:raise RuntimeError('28reserved slots unavailable; no GPUqueued')
    job=subprocess.check_output(['sbatch','--parsable',f'--dependency=afterok:{os.environ["SLURM_JOB_ID"]}','pilot.sbatch'],cwd=root,text=True).strip().split(';')[0]
    (root/'continuation.json').write_text(json.dumps(dict(source_pilot_job=job)))
    print(f'325 CPU real no-score guard passed; one A100sourcepilot {job}',flush=True)


if __name__=='__main__':main()
