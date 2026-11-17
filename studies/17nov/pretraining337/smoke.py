"""Real source pretext/embedding cost; no native40 fit or held scores."""
import json
import os
import time
from pathlib import Path
import numpy as np
import torch
from common import source,native,capture,embedding,sha,ARMS,state_sha
from pretext import attach

root=Path(os.environ['PRETRAIN_ROOT']);torch.set_num_threads(6);start=time.monotonic()
path=Path(os.environ['TEACHER_ROOT']);card=json.loads((path/'teacher.json').read_text())
if sha(path/'teacher.npz')!=card['teacher_sha256']:raise ValueError('teacher pin')
with np.load(path/'teacher.npz',allow_pickle=False) as z:teacher={k:z[k] for k in z.files}
records={}
for arm in ARMS:
    d,fit,held,people,y=source();model=native(d);capture(model,fit)
    record=attach(model,arm,teacher,fit,steps=8);model.fit(fit)
    tick=time.monotonic();embedding(model,fit[:32]);cost=time.monotonic()-tick
    if record['steps']!=8 or not record['fine_rng_restored']:raise ValueError('pretext/restore noop')
    records[arm]=dict(pretext8_seconds=record['pretext_seconds'],embedding32_seconds=cost,losses=record['losses'],body_change=record['body_change_max'])
# Native40 conservative cost comes from independent source6 guard338, not a weakened fit.
future=sum(6*(488.647+1.5*v['pretext8_seconds']*8+1.5*v['embedding32_seconds']*(len(fit)+len(held))/32+30) for v in records.values())
used=266+2*card['seconds']+6*(time.monotonic()-start)
projection=future+used+600 # collector/reconstruction resource reserve
report=dict(passed=True,budget_passed=projection<=18000,projected_total_coreseconds=projection,records=records,
    source_sha256=sha(root/'source.sha256'),teacher_sha256=card['teacher_sha256'],seconds=time.monotonic()-start,held_labels_opened=False,P02P03_inference=False)
(root/'smoke.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
if not report['budget_passed']:raise ValueError('5CPUh budget stops before full study')
