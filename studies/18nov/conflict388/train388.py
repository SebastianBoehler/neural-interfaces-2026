"""Compare paired gradient projections at identical source episodes and optimizer budgets."""
import hashlib
from itertools import combinations
import json
from pathlib import Path
import sys
import numpy as np
import torch
from torch.nn import functional as F

BUNDLE=Path.home()/'ni26/382/bundle'
sys.path.insert(0,str(BUNDLE))
from kernel import configure,group_digest,sha_array
BUNDLE_SHA='06f45ef836fab01845cbb007bb5567dc1c39b4835b1b84b4930fdac1f4d6545f'
PEOPLE=('P04','P05','P06','P07','P08','P10')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def schedule(streams,seed):
    rng=np.random.default_rng(seed)
    grp=np.array([s.rsplit('/',1)[1].split('|')[0] for s in streams])
    assert tuple(np.unique(grp))==PEOPLE
    pairs=list(combinations(PEOPLE,2))
    out=[]
    for _ in range(8):
        for idx in rng.permutation(len(pairs)):
            pair=pairs[idx]
            batch=[]
            for p in pair:
                stream=rng.choice(np.unique(streams[grp==p]))
                batch.append(rng.choice(np.flatnonzero(streams==stream),48,replace=False))
            out.append(batch)
    return np.array(out)


def main(arm,seed,smoke):
    assert arm in 'NHX' and seed in (382,1382) and 'H100' in torch.cuda.get_device_name()
    meta=json.loads((BUNDLE/'bundle.json').read_text())
    assert sha(BUNDLE/'bundle.json')==BUNDLE_SHA
    for p,h in meta['files'].items():
        assert sha(BUNDLE/p)==h
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    net=torch.jit.load(str(BUNDLE/'learned.ts'),map_location='cuda').eval()
    groups=configure(net,'J')
    params=[p for ps in groups.values() for p in ps]
    frozen=[p for p in net.parameters() if not p.requires_grad]
    original={g:group_digest(ps) for g,ps in groups.items()}
    fixed=group_digest(frozen)
    optimizer=torch.optim.AdamW([dict(params=ps,lr=.0003 if k=='norm' else .00003) for k,ps in groups.items()],weight_decay=.001)
    with np.load(BUNDLE/'source.npz') as d:
        x,y,w=[torch.from_numpy(d[k]).cuda() for k in ('x','y','weights')]
        ss=d['streams']
    from kernel import schedule as base_schedule
    probe=base_schedule(ss,382,1)[0]
    with torch.no_grad(),np.load(BUNDLE/'probes.npz') as z:
        baseline=torch.from_numpy(z['learned']).cuda()
        actual=net(x[probe])
        assert (actual-baseline).abs().max()<.005 and torch.equal(actual.argmax(1),baseline.argmax(1))
    episodes=schedule(ss,seed)
    conflicts=0
    max_update=0.
    hard_weights=[]
    grads={}
    for pair in episodes[:15 if smoke else 120]:
        loss=[]
        gs=[]
        for b in pair:
            pred=net(x[b])
            ce=(F.cross_entropy(pred,y[b],reduction='none')*w[b]).mean()
            assert torch.isfinite(ce)
            grad=torch.autograd.grad(ce,params)
            gs.append(torch.cat([t.reshape(-1) for t in grad]))
            loss.append(ce.detach())
        a,b=gs
        hard=torch.stack(loss).softmax(0)
        assert int(torch.stack(loss).argmax())==int(hard.argmax())
        hard_weights.append(float(hard.max()))
        weight=.5 if arm=='N' else hard[0]
        raw=weight*a+(1-weight)*b
        dot=torch.dot(a,b)
        if float(dot)<0:
            conflicts+=1
            if arm=='X':
                left=a-dot*b/torch.dot(b,b)
                right=b-dot*a/torch.dot(a,a)
                update=weight*left+(1-weight)*right
                max_update=max(max_update,float((raw-update).abs().max()))
            else:
                update=raw
        else:
            update=raw
        assert torch.isfinite(update).all()
        optimizer.zero_grad()
        for p,g in zip(params,update.split([p.numel() for p in params])):
            p.grad=g.reshape_as(p)
        if not grads:
            grads={k:max(float(p.grad.abs().max()) for p in ps) for k,ps in groups.items()}
            assert all(v>0 for v in grads.values())
        optimizer.step()
    after={k:group_digest(ps) for k,ps in groups.items()}
    assert all(original[k]!=after[k] for k in groups) and group_digest(frozen)==fixed
    assert conflicts>0 and max(hard_weights)>.5 and (arm!='X' or max_update>0)
    key=('smoke_' if smoke else '')+arm+'_'+str(seed)
    record=dict(arm=arm,seed=seed,episodes=sha_array(episodes),steps=15 if smoke else 120,bundle=BUNDLE_SHA,
                source=sha(Path('source.sha256')),people=PEOPLE,rows=meta['source_rows_sha256'],
                original=original,after=after,frozen=fixed,first_grad=grads,conflicts=conflicts,
                max_projection_delta=max_update,mean_higher_loss_weight=float(np.mean(hard_weights)),
                held_labels_opened=False,device=torch.cuda.get_device_name())
    if not smoke:
        path=Path(key+'.pt')
        torch.save({k:p.detach().cpu() for k,p in net.state_dict().items()},path)
        record['weights']=sha(path)
    Path(key+'.json').write_text(json.dumps(record,indent=2))
    print(key,conflicts,max_update,flush=True)


if __name__=='__main__':
    main(sys.argv[1],int(sys.argv[2]),sys.argv[3:]==['smoke'])
