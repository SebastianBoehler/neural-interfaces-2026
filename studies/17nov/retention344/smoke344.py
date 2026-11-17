"""Real source gradients/cost and exact native/zeroaux update twin, no query metrics."""
import json
import os
import time
from pathlib import Path
import numpy as np
import torch
from common344 import teacher_load,sha
from run344 import fit_arm

root=Path(os.environ['RETENTION_ROOT']);torch.set_num_threads(6);teacher,card=teacher_load();start=time.monotonic();records={}
fit_arm('A',teacher,root,limit=2,save=False) # source-only library/optimizer warm-up, reset native seed afterward
for arm in ('A','Z','B','C','D'):
    model,d,fit,held,y,report=fit_arm(arm,teacher,root,limit=8,save=False,presteps=8)
    records[arm]=report
if records['A']['body_sha256']!=records['Z']['body_sha256'] or records['A']['concurrent']['torch_rng_after']!=records['Z']['concurrent']['torch_rng_after'] or records['A']['concurrent']['batch_hashes']!=records['Z']['concurrent']['batch_hashes']:raise ValueError('zeroaux native update/RNG identity')
for arm in ('C','D'):
    r=records[arm]['concurrent']
    if r['actual_steps']!=8 or r['steps']!=8 or not np.isclose(r['alpha']*r['aux_gradient_norm']/r['task_gradient_norm'],.1):raise ValueError('active source auxiliary')
if records['C']['concurrent']['alpha']!=records['D']['concurrent']['alpha']:raise ValueError('C/D fixed lambda mismatch')
# Historical native upper488s, measured auxiliary perstep upper1.5x + snapshots/inference90s each.
future=6*sum(max(488.,1.5*r['concurrent']['fine_seconds']/8*1240)+1.5*r.get('pretext',{}).get('pretext_seconds',0)*8+90 for a,r in records.items() if a!='Z')
used=108+354+6*(time.monotonic()-start);reserve=1800
projection=future+used+reserve
report=dict(passed=True,budget_passed=projection<=28800,projected_coreseconds=projection,seconds=time.monotonic()-start,records=records,source_sha256=sha(root/'source.sha256'),teacher_sha256=card['teacher_sha256'],held_labels_opened=False)
(root/'smoke.json').write_text(json.dumps(report));print(json.dumps({k:v for k,v in report.items() if k!='records'}),flush=True)
if not report['budget_passed']:raise ValueError('8CPUh projection stop')
