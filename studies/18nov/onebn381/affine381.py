"""Measure transferred normalization accuracy apart from learned task decisions."""
import json
from pathlib import Path
import sys
import numpy as np
import torch

ACTOR=Path('/mnt/lustre/work/gehler/gfh527/ni26_onebn381_actor_v1_20261118')
sys.path.insert(0,str(ACTOR))
from evaluate import TRAIN, owners, joint, source, preactivation, LearnedBN
from train import teacher


@torch.no_grad()
def main():
    d,people,rows,runtime=source()
    assert 'y' not in d
    joint.check_manifest(ACTOR,'1c6cd0e0a399cc60979801982437d5614b1e7635ec7f8899a80106401bbd5aaf')
    fits=owners(rows)
    net=runtime.net
    original=net.spatial_conv[1]
    models={}
    for name in ('C','D','DC'):
        model=LearnedBN(original)
        model.load_state_dict(torch.load(TRAIN/(name+'.pt'),map_location='cpu',weights_only=True))
        models[name]=model.eval()
    # Use identical pointwise input for all operators before investigating the affine discrepancy.
    x=torch.from_numpy(runtime._aligned(np.arange(len(people)), True))
    hidden=torch.cat([preactivation(net,xx) for xx in x.split(32)])
    perm=joint.permutations(len(people))
    streams=[rows[d['stream'][rows]==s] for s in sorted(set(d['stream'][rows]))]
    rng=np.random.default_rng(1381)
    episodes=[(f'source{i}',rng.choice(streams[rng.integers(len(streams))],48,replace=False)) for i in range(24)]
    for p in joint.COHORT:
        pos=np.flatnonzero(people==p)
        assert len(set(d['stream'][pos]))==1
        episodes.extend((p+'|'+o,v[np.isin(v,pos)]) for o,v in perm.items())
    results={}
    for tag,ids in episodes:
        h=hidden[ids]
        desired=torch.stack(teacher(h,original),dim=-1)
        losses={'static':float(desired.square().mean())}
        for name,layer in models.items():
            predicted=torch.stack(layer.affine(layer.encode(layer.standardized(h)),'real'),dim=-1)
            losses[name]=float((predicted-desired).square().mean())
        results[tag]=losses
    all_groups={}
    for name,keys in [('source',[a for a in results if a.startswith('source')])]+[
                        (p,[p+'|'+o for o in joint.ORDERS]) for p in joint.COHORT]:
        def summary(arm):
            v=np.array([results[k][arm] for k in keys])
            return float(v.mean() if name=='source' else (v[:4].mean()+v[4:6].mean()+v[6:].mean())/3)
        all_groups[name]={a:summary(a) for a in ('static','C','D','DC')}
    all_groups['held']={a:np.mean([all_groups[p][a] for p in joint.COHORT]) for a in all_groups['source']}
    output=dict(mse=all_groups,source_sha=joint.sha(Path('source.sha256')),held_labels_opened=False,
                teacher='original single k20 layer',scope='log-gain and additive shift, equal weight')
    Path('result.json').write_text(json.dumps(output,indent=2))
    print(all_groups,flush=True)


if __name__=='__main__':
    main()
