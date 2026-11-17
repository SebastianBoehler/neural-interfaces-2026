"""Fresh session calls, deterministic source donor, complete pinned-order outputs."""
import copy
import numpy as np
from source332 import modes,fingerprint,ORDERS
from views332 import prepare


def probability(model,rows,donor,d,scale,view,device):
    x=d['X'][rows]/scale
    if view=='D':x=np.concatenate([x,d['X'][np.resize(donor,len(rows))]/scale])
    value=modes.probabilities(model,x,device)
    return value[:len(rows)]


def run(d,test,donor,perms,nets,original,device,check_original=True,orders=ORDERS):
    position=np.full(len(d['X']),-1,int);position[test]=np.arange(len(test))
    outputs={};checks={};donors={}
    for label,(net,scale) in nets.items():
        probe=d['X'][test[:min(16,len(test))]]/scale
        import torch
        views,counts=prepare(net,torch.as_tensor(probe,device=device).float())
        for view,model in views.items():
            state=fingerprint(model);maximum=endpoint=0.;effect=0.
            for order in orders:
                p=perms[order]
                if not np.array_equal(np.sort(p),np.arange(len(d['X']))):raise ValueError('badpinnedpermutation')
                rank=np.empty(len(p),int);rank[p]=np.arange(len(p))
                source=donor[np.argsort(rank[donor],kind='stable')]
                donors[order]=source.tolist();arr=np.full((len(test),3),np.nan)
                for session in dict.fromkeys(d['stream'][test]):
                    rows=test[d['stream'][test]==session];rows=rows[np.argsort(rank[rows],kind='stable')]
                    q=probability(model,rows,source,d,scale,view,device)
                    n=min(10,len(rows));prefix=probability(model,rows[:n],source,d,scale,view,device)
                    error=float(np.max(np.abs(q[:n]-prefix)));maximum=max(maximum,error)
                    if not np.allclose(q[:n],prefix,atol=2e-5,rtol=2e-5):raise ValueError('targetprefix/futureguard')
                    other=test[d['stream'][test]!=session]
                    if len(other):probability(model,other[:n],source,d,scale,view,device)
                    repeated=probability(model,rows[:n],source,d,scale,view,device)
                    copied=probability(copy.deepcopy(model),rows[:n],source,d,scale,view,device)
                    if not np.array_equal(prefix,repeated) or not np.array_equal(prefix,copied):raise ValueError('reset/interleave/deepcopy')
                    arr[position[rows]]=q
                if not np.isfinite(arr).all() or not np.allclose(arr.sum(1),1,atol=1e-6):raise ValueError('invalidviewbank')
                if view in ('F','B') and check_original:
                    old=original[f'{label}{"C" if view=="B" else "F"}|{order}']
                    endpoint=max(endpoint,float(np.max(np.abs(arr-old))))
                    if not np.allclose(arr,old,atol=2e-5,rtol=2e-5) or not np.array_equal(arr.argmax(1),old.argmax(1)):
                        raise ValueError(f'originalendpointidentityfailed:{label}{view}/{order},max={endpoint}')
                outputs[f'{label}{view}|{order}']=arr
            if fingerprint(model)!=state:raise ValueError('viewmutatedfittedparameters/buffers')
            checks[label+view]=dict(prefix_max_abs=maximum,original_endpoint_max_abs=endpoint,
                state_sha256=state,BN_layers=counts.get(view,sum(1 for m in model.body.modules() if hasattr(m,'running_mean'))))
        for view in ('M','V','D'):
            effect=max(float(np.max(np.abs(outputs[f'{label}{view}|{o}']-outputs[f'{label}F|{o}']))) for o in orders)
            checks[label+view]['effect_vs_fixed_max_abs']=effect
            if effect<=1e-7:raise ValueError(f'interventiongloballyinactive:{label}{view}:{effect}')
    return outputs,checks,donors
