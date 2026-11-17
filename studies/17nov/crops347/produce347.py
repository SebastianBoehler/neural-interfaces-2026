"""Two source-only matched full fits; all held banks saved without held labels."""
import json
import os
import sys
import time
from pathlib import Path
import numpy as np
import torch
from training347 import run,sha,digest,state_sha,CONTROL
from common346 import bank,permutations,ORDERS

SMOKE=Path('/mnt/lustre/work/gehler/gfh527/ni26_crops347_train_smoke_v1_20261117')


def main():
    root=Path(os.environ['TRAIN347_ROOT']);start=time.monotonic();torch.set_num_threads(6);arm=('B','C')[int(sys.argv[1])]
    entry=json.loads((root/'preflight.json').read_text())
    if not entry['passed'] or entry['source_sha256']!=sha(root/'source.sha256') or entry['held_labels_opened'] or entry['new_fits']!=0:raise ValueError('actual same-source entrypoint/control preflight required')
    smoke=json.loads((SMOKE/'smoke.json').read_text())
    if not smoke['passed'] or not smoke['budget_passed'] or not smoke['original_zero_BITWISE'] or smoke['source_sha256']!=sha(SMOKE/'source.sha256'):raise ValueError('passed same-method cost/zero smoke required')
    if sha(root/'training347.py')!=sha(SMOKE/'training347.py'):raise ValueError('scientific method differs from actual smoke')
    d,fit,_,people,yfit,model,trace,qualification,seconds=run(arm)
    control=json.loads((CONTROL/'record.json').read_text())
    if trace['steps']!=2480 or trace['epochs']!=40 or trace['base_batches']!=control['trace']['batches'] or len(trace['extra_batches'])!=40:raise ValueError('complete original31base/31extra×40 source recipe failed')
    query=np.flatnonzero(np.isin(people,['P02','P03','P09']));perms=permutations(len(people))
    if np.intersect1d(fit,query).size or np.any(d['y'][query]!=-1):raise ValueError('query leaked into source fit')
    prediction={};checks={}
    for order,perm in perms.items():_,prediction[order],checks[order]=bank(model,d,query,perm)
    # Source training quality diagnostic only; never selects duration or a query arm.
    _,source_p,source_guard=bank(model,d,fit,perms['within0'])
    quality=dict(pooledNLL=float(-np.log(np.maximum(source_p[np.arange(len(yfit)),yfit],1e-12)).mean()),
        class_recalls=[float(np.mean(source_p[yfit==c].argmax(1)==c)) for c in range(3)])
    folder=root/'output'/arm;folder.mkdir(parents=True,exist_ok=False)
    torch.save(dict(state=model.net.state_dict(),G=model.G,scale=model.scale),folder/'model.pt')
    np.savez(folder/'predictions.npz',index=query,**prediction)
    record=dict(arm=arm,source_sha256=sha(root/'source.sha256'),method_sha256=sha(root/'training347.py'),
        smoke_sha256=sha(SMOKE/'smoke.json'),qualification_source_sha256=qualification['source_sha256'],
        late_sha256=qualification['late_sha256'],checkpoint_sha256=sha(folder/'model.pt'),prediction_sha256=sha(folder/'predictions.npz'),
        source_rows_sha256=digest(fit),query_rows_sha256=digest(query),source_people=sorted(set(people[fit])),
        query_people=sorted(set(people[query])),trace=trace,guards=checks,source_guard=source_guard,source_quality=quality,
        final_state_sha256=state_sha(model.net),held_labels_opened=False,training_seconds=seconds,seconds=time.monotonic()-start)
    (folder/'record.json').write_text(json.dumps(record,indent=2));print(json.dumps({k:v for k,v in record.items() if k not in ('trace','guards','source_guard')}),flush=True)


if __name__=='__main__':main()
