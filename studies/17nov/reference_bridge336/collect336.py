"""Onlycompletefivearmbridge mayopenexposedquerylabels; originalmetriconly."""
import json
import os
from pathlib import Path
import numpy as np
from recipe336 import source,labels,sha,digest,official
from produce336 import ARMS

ORDERS=('within0','within1','within2','within3','full0','full1','runs0','runs1')


def main():
    root=Path(os.environ['BRIDGE_ROOT']);d,fit,query,people,yfit=source();records={};banks={};hashes={}
    if {p.name for p in (root/'output').iterdir()}!=set(ARMS):raise ValueError('complete5armsrequired')
    for arm in ARMS:
        folder=root/'output'/arm;m=json.loads((folder/'record.json').read_text())
        if m['arm']!=arm or m['query_labels_opened'] or m['device']!='cpu' or m['source_sha256']!=sha(root/'source.sha256'):raise ValueError('armpin')
        if m['checkpoint_sha256']!=sha(folder/'model.pt') or m['source_rows_sha256']!=digest(fit) or m['query_rows_sha256']!=digest(query):raise ValueError('checkpoint/rowchanged')
        if m['trace']['hyper']['epochs']!=40 or len(m['trace']['batches'])!=40 or not m['trace']['optimizer_owns_actual_head']:raise ValueError('recipe/no-opguard')
        with np.load(folder/'predictions.npz') as z:
            if set(z.files)!={'index'}|set(ORDERS) or not np.array_equal(z['index'],query):raise ValueError('querybankindices/orders')
            banks[arm]={o:z[o].copy() for o in ORDERS}
            if any(p.shape!=(len(query),3) or not np.isfinite(p).all() or not np.allclose(p.sum(1),1,atol=1e-6) for p in banks[arm].values()):raise ValueError('badpredictions')
        records[arm]=m;hashes[arm]=sha(folder/'predictions.npz')
    anchor=records['A']['trace']
    for arm in ARMS:
        trace=records[arm]['trace']
        if trace['initial_state_sha256']!=anchor['initial_state_sha256']:raise ValueError('initializationdiffers')
        for key in ('torch_rng_before','torch_rng_after','augment_rng_before','augment_rng_after'):
            if trace[key]!=anchor[key]:raise ValueError('RNGschedulecoupling:'+arm+'/'+key)
        if arm!='B' and trace['batches']!=anchor['batches']:raise ValueError('unintendedbatchchange')
        if (arm=='H')==trace['head_is_parametrized']:raise ValueError('headconstrainttogglewrong')
        if arm!='U' and trace['scale']!=anchor['scale']:raise ValueError('unintendedscalechange')
        if arm!='W' and trace['criterion_weights']!=anchor['criterion_weights']:raise ValueError('unintendedlosschange')
    metric=official();y=labels(query,len(d['X']));scores={}
    for arm,values in banks.items():
        result={}
        for person in ('P02','P03'):
            selected=people[query]==person;mask=people==person
            by={o:metric.score(y[selected],p[selected].argmax(1),d,mask)['balanced_accuracy_cell_mean'] for o,p in values.items()}
            result[person]=dict(by_order=by,scheme_equal=float(np.mean([np.mean([v for o,v in by.items() if o.startswith(s)]) for s in ('within','full','runs')])))
        scores[arm]=dict(people=result,person_mean=float(np.mean([v['scheme_equal'] for v in result.values()])))
    report=dict(scope='selectedsource7onefactorbridge;notarchitecture/outer/chaingain',source_sha256=sha(root/'source.sha256'),
        scores=scores,records=records,prediction_sha256=hashes,official_metric_sha256=sha(metric.__file__),no_automatic_expansion=True)
    with (root/'results.json').open('x') as out:json.dump(report,out,indent=2)
    print('336complete5arm/nativeinitialization/RNG/onefactor/actualhead/sourceguards beforequerymetric',flush=True)


if __name__=='__main__':main()
