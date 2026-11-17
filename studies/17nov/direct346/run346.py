"""Complete matched whole-readout suite before opening exposed P09 labels."""
import json
import os
import time
from pathlib import Path
import numpy as np
import torch
from common346 import source,restored,permutations,bank,sha,digest,BASE,CONTROL,ORDERS,labels,official
from features346 import direct,shuffled,fit_head,predict


def main():
    root=Path(os.environ['DIRECT_ROOT']);start=time.monotonic();torch.set_num_threads(6)
    smoke=json.loads((root/'smoke.json').read_text())
    if not smoke['passed'] or not smoke['budget_passed'] or smoke['source_sha256']!=sha(root/'source.sha256'):raise ValueError('qualified same-source smoke required')
    d,fit,held,people,yfit=source();model,card=restored(d,fit,held);perms=permutations(len(people))
    source_latent,_,source_guard=bank(model,d,fit,perms['within0']);source_extra=direct(d['X'][fit])
    with np.load(BASE/'data/proteus_windows.npz') as z:run=z['run'].copy()
    deranged,order=shuffled(fit,d,run,source_extra)
    source_x={'B':source_latent,'C':np.c_[source_latent,source_extra],'D':np.c_[source_latent,deranged]}
    heads={};headarrays={};records={}
    for arm,x in source_x.items():
        heads[arm]=fit_head(x,yfit);scale,head,record=heads[arm];records[arm]=record
        if arm!='B' and np.max(np.abs(head.coef_[:,1152:]))<=1e-10:raise ValueError('extra coefficients inactive')
        headarrays.update({arm+'_mean':scale.mean_,arm+'_scale':scale.scale_,arm+'_coef':head.coef_,arm+'_intercept':head.intercept_})
    # Heads/scalers are finalized before even transforming held EEG.
    np.savez(root/'heads.npz',**headarrays)
    (root/'heads_before_query.json').write_text(json.dumps(dict(source_sha256=sha(root/'source.sha256'),
        source_people=sorted(set(people[fit])),source_rows_sha256=digest(fit),source_native=source_guard,
        source_input_sha256=digest(d['X'][fit]),derangement_sha256=digest(order),heads=records,
        checkpoint_sha256=card['checkpoint_sha256'],held_transformed=False,held_labels_opened=False)))
    extra=direct(d['X'][held]);banks={a:{} for a in ('A','B','C','D')};guards={};identity={}
    with np.load(CONTROL/'predictions.npz') as original:
        if not np.array_equal(original['index'],held):raise ValueError('strong control query rows')
        for name,perm in perms.items():
            latent,p,guard=bank(model,d,held,perm);error=float(np.max(np.abs(p-original[name])))
            if error>1e-6 or not np.array_equal(p.argmax(1),original[name].argmax(1)):raise ValueError('actualA full original identity failed')
            identity[name]=error;guards[name]=guard;banks['A'][name]=p
            for arm,head in heads.items():banks[arm][name]=predict(head,latent if arm=='B' else np.c_[latent,extra])
    if any(all(np.array_equal(banks[a][o],banks['B'][o]) for o in ORDERS) for a in ('C','D')):raise ValueError('direct prediction silent no-op')
    np.savez(root/'predictions.npz',index=held,**{a+'_'+o:p for a,bb in banks.items() for o,p in bb.items()})
    before=dict(all_four_banks_complete=True,source_sha256=sha(root/'source.sha256'),heads_sha256=sha(root/'heads.npz'),
        source_proof_sha256=sha(root/'heads_before_query.json'),prediction_sha256=sha(root/'predictions.npz'),
        identity=identity,guards=guards,held_labels_opened=False)
    (root/'before_labels.json').write_text(json.dumps(before,indent=2))
    y=labels(held,len(people));metric=official();scores={}
    def scheme(b):return float(np.mean([np.mean([b[o] for o in ORDERS if o.startswith(s)]) for s in ('within','full','runs')]))
    for arm,bb in banks.items():
        by={o:metric.score(y,p.argmax(1),d,people=='P09')['balanced_accuracy_cell_mean'] for o,p in bb.items()}
        nll={o:float(-np.log(np.maximum(p[np.arange(len(y)),y],1e-12)).mean()) for o,p in bb.items()}
        scores[arm]=dict(accuracy=scheme(by),by_order=by,NLL=scheme(nll))
    result=dict(source_sha256=sha(root/'source.sha256'),scores=scores,
        delta_pp={a+'-'+b:100*(scores[a]['accuracy']-scores[b]['accuracy']) for a,b in [('C','B'),('C','D'),('C','A'),('B','A')]},
        before_labels_sha256=sha(root/'before_labels.json'),metric_sha256=sha(metric.__file__),
        heads=records,seconds=time.monotonic()-start,scope='INDIVIDUAL-MODEL exposedP09/source6/seed0;no FULL-SYSTEM DEV/independence/familyclosure')
    (root/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)


if __name__=='__main__':main()
