"""Source-only teacher cost and native zero/activity guards before held labels."""
import json
import os
import time
from pathlib import Path
import numpy as np
import torch
from common345 import source,native,observe,state_sha,sha,digest,permutations
from teacher345 import caches,fit_predict

root=Path(os.environ['DISTILL_ROOT']);torch.set_num_threads(6);started=time.monotonic()
d,fit,held,people,y=source();v,hashes=caches(d,people);perms=permutations(len(people))
p,teacher=fit_predict('P04',v,d,people,y,perms,smoke=False)
results={};models={}
# Exact existing native implementation: zero ignores teacher, active uses a source-only teacher.
for arm in ('plain','zero','active'):
    model=native(d,epochs=1);model.dist=.5 if arm=='active' else 0.
    model.dist_T=2.;model.teacher=(fit,np.tile(p['within0'][0],(len(fit),1)))
    trace=observe(model,'A',fit);tick=time.monotonic();model.fit(fit)
    results[arm]=dict(state_sha256=state_sha(model.net),trace=trace,seconds=time.monotonic()-tick,dg=model.dg_stats)
    models[arm]=model
if results['plain']['state_sha256']!=results['zero']['state_sha256'] or results['plain']['trace']!=results['zero']['trace']:
    # Wall-clock trace fields are measurements, not numerical identity.
    a=results['plain']['trace'];b=results['zero']['trace']
    aa={k:v for k,v in a.items() if k!='full_step_seconds'};bb={k:v for k,v in b.items() if k!='full_step_seconds'}
    if results['plain']['state_sha256']!=results['zero']['state_sha256'] or aa!=bb:raise ValueError('zero KD is not exact native')
active=results['active']
if active['state_sha256']==results['plain']['state_sha256'] or active['dg']['dist_steps']!=active['trace']['steps'] or active['dg']['dist_kl']<=0:raise ValueError('KD inactive')
projection=6*(14*teacher['seconds']+2*max(489.,results['active']['seconds']*40)*1.25+time.monotonic()-started+120)+1800
record=dict(passed=True,budget_passed=projection<=14400,projected_coreseconds=projection,source_sha256=sha(root/'source.sha256'),
    teacher_smoke=teacher,cache_hashes=hashes,results=results,held_labels_opened=False,seconds=time.monotonic()-started)
(root/'smoke.json').write_text(json.dumps(record,indent=2));print(json.dumps(record),flush=True)
if not record['budget_passed']:raise ValueError('4CPUcoreh budget stops before science')
