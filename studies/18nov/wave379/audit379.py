"""Verify all source episodes, trained checkpoints and immediate CPU decisions independently."""
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import torch
from model import Decoder
from score379 import DATA_SHA, ORDER_SHA, ORDERS, PEOPLE, sha


@torch.no_grad()
def replay(model,features,arm,donor):
    # One call per window. Reset the three accumulators at the start of each session.
    first=torch.zeros(48)
    second=torch.zeros(48)
    code=torch.zeros(48)
    labels=[]
    for n,z in enumerate(features):
        center=model.mean if arm!='norm' else (20*model.mean+first)/(20+n)
        square=model.second if arm!='norm' else (20*model.second+second)/(20+n)
        learned=arm in ('memory','donor')
        history=torch.cat((code/max(1,n),torch.tensor([n/(n+20)]))) if learned else torch.zeros(49)
        bias,gain=model.read(history).chunk(2)
        norm=(z-center[:,None])*(square-center.square()).clamp_min(1e-6).rsqrt()[:,None]
        transformed=norm*gain.tanh().exp()[:,None]+bias[:,None]
        block=transformed.reshape(48,4,-1).square().mean(-1).clamp_min(1e-8).log().flatten()
        labels.append(model.head(block).argmax().item())
        source=donor[n%len(donor)] if arm=='donor' else z
        mu=source.mean(-1)
        power=source.square().mean(-1)
        first+=mu
        second+=power
        code+=model.write(torch.cat((mu,power.clamp_min(1e-8).log())))
    return np.array(labels)


@torch.no_grad()
def main(root,data,perm,output):
    torch.set_num_threads(2)
    assert sha(data)==DATA_SHA and sha(perm)==ORDER_SHA
    with np.load(data) as z:
        # The CPU audit opens only EEG and grouping metadata.
        raw=torch.from_numpy(z['X']).float()
        ids=np.array([p.split('/')[-1] for p in z['subject']])
        streams=np.char.add(np.char.add(ids,'|'),z['session'])
    with np.load(perm) as z:
        orders={o:z[f'{o[:-1]}|{o[-1]}|all_loso'] for o in ORDERS}
    diffs=[]
    count=0
    for p in PEOPLE:
        fit=np.flatnonzero(ids!=p)
        test=np.flatnonzero(ids==p)
        sessions=[fit[streams[fit]==s] for s in sorted(set(streams[fit]))]
        rng=np.random.default_rng(379)
        batches=np.stack([rng.choice(sessions[rng.integers(len(sessions))],96,replace=False) for _ in range(360)])
        m=json.loads((root/(p+'.json')).read_text())
        assert m['source_episodes']==hashlib.sha256(batches.tobytes()).hexdigest()
        scale=raw[fit].square().mean().sqrt()
        with np.load(root/(p+'.npz')) as out:
            for parent in ('static','norm','memory'):
                net=Decoder().eval()
                checkpoint=root/f'{p}_{parent}.pt'
                assert sha(checkpoint)==m['checkpoints'][parent]
                net.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=True))
                feats=net.encode(raw[test]/scale)
                donor=net.encode(raw[sessions[0]]/scale)
                for o,order in orders.items():
                    for s in np.unique(streams[test]):
                        rows=order[np.isin(order,test[streams[test]==s])]
                        pos=np.searchsorted(test,rows)
                        for arm in ((parent,'blank','donor') if parent=='memory' else (parent,)):
                            labels=replay(net,feats[pos],arm,donor)
                            bad=np.flatnonzero(labels!=out[arm+'|'+o][pos])
                            count+=len(pos)
                            if len(bad):
                                diffs.append(dict(person=p,order=o,session=s,arm=arm,rows=rows[bad].tolist()))
    output.write_text(json.dumps(dict(checked_episodes=3600,checkpoints=30,decisions=count,
        differences=diffs,data_sha=DATA_SHA,orders_sha=ORDER_SHA,held_labels_opened=False),indent=2))


if __name__=='__main__':
    main(*(Path(p) for p in sys.argv[1:]))
