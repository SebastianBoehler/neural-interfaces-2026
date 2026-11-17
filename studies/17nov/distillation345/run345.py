"""Native40 task distillation; saved complete eight-order predictions, no query labels."""
import json
import os
import sys
import time
from pathlib import Path
import numpy as np
import torch
from common345 import source,native,observe,inference,sha,digest,CONTROL

root=Path(os.environ['DISTILL_ROOT']);torch.set_num_threads(6);start=time.monotonic();arm=('KD','deranged')[int(sys.argv[1])]
smoke=json.loads((Path(os.environ['SMOKE_ROOT'])/'smoke.json').read_text())
if not smoke['passed'] or not smoke['budget_passed']:raise ValueError('smoke/cost gate')
troot=Path(os.environ['TEACHER_ROOT']);card=json.loads((troot/'teacher.json').read_text())
if sha(troot/'teacher.npz')!=card['teacher_sha256']:raise ValueError('teacher changed')
d,fit,held,people,y=source()
with np.load(troot/'teacher.npz') as z:
    if not np.array_equal(z['index'],fit) or not np.array_equal(z['query'],held):raise ValueError('teacher scope')
    teacher=z['proba' if arm=='KD' else 'deranged'].copy()
if not np.isfinite(teacher).all() or not np.allclose(teacher.sum(1),1):raise ValueError('invalid teacher')
model=native(d);model.dist=.5;model.dist_T=2.;model.teacher=(fit,teacher);trace=observe(model,'A',fit);model.fit(fit);model.set_k(20.)
control=json.loads((CONTROL/'record.json').read_text());reference=control['trace']
for key in ('initial_state_sha256','scale','steps','batches','torch_rng_before','torch_rng_after','augment_rng_before','augment_rng_after'):
    if trace[key]!=reference[key]:raise ValueError('native matched recipe:'+key)
if trace['steps']!=1240 or model.dg_stats['dist_steps']!=1240 or model.dg_stats['dist_kl']<=0:raise ValueError('native40/KD activity')
p,checks=inference(model,d,fit,held,y);folder=root/'output'/arm;folder.mkdir(parents=True,exist_ok=False)
checkpoint=folder/'model.pt';torch.save(dict(state=model.net.state_dict(),G=model.G,scale=model.scale),checkpoint)
with (folder/'predictions.npz').open('xb') as f:np.savez(f,index=held,**p)
record=dict(arm=arm,source_sha256=sha(root/'source.sha256'),teacher_sha256=card['teacher_sha256'],checkpoint_sha256=sha(checkpoint),
    prediction_sha256=sha(folder/'predictions.npz'),source_rows_sha256=digest(fit),held_rows_sha256=digest(held),trace=trace,
    dg=model.dg_stats,inference=checks,held_labels_opened=False,P02P03_inference=False,seconds=time.monotonic()-start)
(folder/'record.json').write_text(json.dumps(record));print(arm,record['seconds'],flush=True)
