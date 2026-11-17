"""All source/activity/reset guards before exposed P09 labels and paired scores."""
import json
import os
from pathlib import Path
import numpy as np
from common345 import source,sha,digest,labels,official,ORDERS,CONTROL

root=Path(os.environ['DISTILL_ROOT']);d,fit,held,people,yfit=source();troot=Path(os.environ['TEACHER_ROOT']);card=json.loads((troot/'teacher.json').read_text())
if sha(troot/'teacher.npz')!=card['teacher_sha256']:raise ValueError('teacher changed')
control=json.loads((CONTROL/'record.json').read_text())
if control['checkpoint_sha256']!='eadd5c23c9f7cac239bf78a99993d8d3e47392d24d166e0ede6fce5b0e6ac9f8' or sha(CONTROL/'model.pt')!=control['checkpoint_sha256'] or sha(CONTROL/'predictions.npz')!=control['prediction_sha256'] or control['query_labels_opened'] or control['source_rows_sha256']!=digest(fit) or control['query_rows_sha256']!=digest(held):raise ValueError('matched raw control identity')
banks={};records={'native':control}
for arm in ('native','KD','deranged'):
    folder=CONTROL if arm=='native' else root/'output'/arm
    record=control if arm=='native' else json.loads((folder/'record.json').read_text())
    if arm!='native':
        if record['held_labels_opened'] or record['source_sha256']!=sha(root/'source.sha256') or record['teacher_sha256']!=card['teacher_sha256'] or record['source_rows_sha256']!=digest(fit) or record['held_rows_sha256']!=digest(held):raise ValueError('producer scope')
        if sha(folder/'model.pt')!=record['checkpoint_sha256'] or sha(folder/'predictions.npz')!=record['prediction_sha256']:raise ValueError('producer file changed')
        if record['dg']['dist_steps']!=1240 or record['dg']['dist_kl']<=0:raise ValueError('inactive KD')
        records[arm]=record
    with np.load(folder/'predictions.npz') as z:
        if not np.array_equal(z['index'],held) or set(z.files)!={'index'}|set(ORDERS):raise ValueError('complete eight order bank')
        banks[arm]={o:z[o].copy() for o in ORDERS}
with np.load(troot/'teacher.npz') as z:
    if not np.array_equal(z['query'],held):raise ValueError('teacher query')
    banks['teacher']={o:z[o].copy() for o in ORDERS}
for name,b in banks.items():
    if any(p.shape!=(len(held),3) or not np.isfinite(p).all() or not np.allclose(p.sum(1),1,atol=1e-6) for p in b.values()):raise ValueError('probability layout')
if all(np.array_equal(banks['KD'][o],banks['native'][o]) for o in ORDERS):raise ValueError('KD prediction silent no-op')
(root/'before_labels.json').write_text(json.dumps(dict(all_guards_passed=True,held_labels_opened=False,source_sha256=sha(root/'source.sha256'),teacher_sha256=card['teacher_sha256'])))
y=labels(held,len(people));metric=official();scores={}
def scheme(b):return float((np.mean([b[o] for o in ORDERS[:4]])+np.mean([b[o] for o in ORDERS[4:6]])+np.mean([b[o] for o in ORDERS[6:]]))/3)
for arm,b in banks.items():
    by={o:metric.score(y,p.argmax(1),d,people=='P09')['balanced_accuracy_cell_mean'] for o,p in b.items()}
    nll={o:float(-np.log(np.clip(p[np.arange(len(y)),y],1e-12,1)).mean()) for o,p in b.items()}
    scores[arm]=dict(accuracy=scheme(by),by_order=by,NLL=scheme(nll))
report=dict(source_sha256=sha(root/'source.sha256'),scores=scores,records=records,teacher=card,
    delta_pp={f'{a}-native':100*(scores[a]['accuracy']-scores['native']['accuracy']) for a in ('KD','deranged','teacher')},
    KD_minus_deranged_pp=100*(scores['KD']['accuracy']-scores['deranged']['accuracy']),before_labels_sha256=sha(root/'before_labels.json'),
    scope='INDIVIDUAL-MODEL exposedP09/source6/seed0;not FULL-SYSTEM DEV or independent confirmation')
(root/'result.json').write_text(json.dumps(report));print(json.dumps(scores),flush=True)
