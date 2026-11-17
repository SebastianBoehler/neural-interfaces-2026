"""Complete original/generic/phase-prior counterfactual matrix before any349gold read."""
import json
import os
from pathlib import Path
import numpy as np
from source349 import source,labels,official,ORDERS,sha,digest
from predict349 import PREP

root=Path(os.environ['RECAL349_ROOT']);d,fit,_,people,_=source();query=np.flatnonzero(np.isin(people,['P02','P03','P09']))
proof=json.loads((root/'prediction_proof.json').read_text());prep=json.loads((PREP/'prepare.json').read_text())
if not proof['passed'] or not proof['all6arms_all3people_all8orders'] or proof['query_labels_opened'] or proof['optimizer_steps']!=0 or proof['source_sha256']!=sha(root/'source.sha256'):raise ValueError('complete guarded source-only counterfactual proof required')
if proof['prepare_sha256']!=sha(PREP/'prepare.json') or proof['buffer_sha256']!=prep['buffer_sha256'] or proof['source_rows_sha256']!=digest(fit) or proof['query_rows_sha256']!=digest(query):raise ValueError('source/statistic/query provenance failed')
if proof['predictions_sha256']!=sha(root/'predictions.npz') or proof['original_before_recal_sha256']!=sha(root/'original_before_recal.json'):raise ValueError('matrix/zero-first proof changed')
ARMS=('A','B','C','B_recal','C_mixed_recal','C_official_recal');banks={}
with np.load(root/'predictions.npz') as z:
    if not np.array_equal(z['index'],query) or set(z.files)!={'index'}|{a+'_'+o for a in ARMS for o in ORDERS}:raise ValueError('complete6×8 matrix schema')
    for a in ARMS:banks[a]={o:z[a+'_'+o].copy() for o in ORDERS}
if any(p.shape!=(len(query),3) or not np.isfinite(p).all() or not np.allclose(p.sum(1),1,atol=1e-6) for bb in banks.values() for p in bb.values()):raise ValueError('incomplete/invalid probabilities')
(root/'before_labels.json').write_text(json.dumps(dict(passed=True,all144banks_complete=True,
    source_sha256=sha(root/'source.sha256'),prediction_proof_sha256=sha(root/'prediction_proof.json'),query_labels_opened=False)))
metric=official();scores={}
def scheme(bb):return float(np.mean([np.mean([bb[o] for o in ORDERS if o.startswith(s)]) for s in ('within','full','runs')]))
for person in ('P02','P03','P09'):
    mask=people[query]==person;yy=labels(query[mask],len(people));scores[person]={}
    for arm,bb in banks.items():
        by={o:metric.score(yy,p[mask].argmax(1),d,people==person)['balanced_accuracy_cell_mean'] for o,p in bb.items()}
        nll={o:float(-np.log(np.maximum(p[mask][np.arange(len(yy)),yy],1e-12)).mean()) for o,p in bb.items()}
        scores[person][arm]=dict(accuracy=scheme(by),by_order=by,pooled_windowNLL=scheme(nll))
contrasts=(('C_official_recal','C_mixed_recal'),('C_official_recal','C'),('C_mixed_recal','C'),('B_recal','B'),('C_official_recal','A'))
result=dict(scores=scores,delta_pp={p:{a+'-'+b:100*(v[a]['accuracy']-v[b]['accuracy']) for a,b in contrasts} for p,v in scores.items()},
    source_sha256=sha(root/'source.sha256'),before_labels_sha256=sha(root/'before_labels.json'),proof=proof,
    metric_sha256=sha(metric.__file__),scope='INDIVIDUAL-MODEL exposedP02/P03/P09/frozen347finalweights/source-onlyONEBN;not independent/fullsystem/phasephysiology')
(root/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(dict(scores=scores,delta_pp=result['delta_pp'])),flush=True)
