"""Measure how much within-session task signal the write edit actually removes."""
import json
from pathlib import Path
import sys

import numpy as np
import torch
from actor import ARCH, CHECKPOINTS, KINDS, PROTO_SHA, ROWS, init, joint, prior
from intervene import variants


def centers(writes, labels):
    means=torch.stack([writes[labels==c].double().mean(0) for c in range(3)])
    return means-means.mean(0)


@torch.no_grad()
def main(smoke, edit=variants):
    d, people, rows, runtime=prior.source()
    assert joint.digest(rows)==ROWS and set(people[rows])==set(joint.FIT_PEOPLE)
    prior.owners(rows)
    joint.pin(Path('prototypes.npz'),PROTO_SHA)
    y=joint.labels(rows,len(people))
    sessions=d['stream'][rows]
    selected=np.unique(sessions)
    if smoke:
        selected=selected[:2]
    ids=np.flatnonzero(np.isin(sessions,selected))
    sessions=sessions[ids]
    y=y[ids]
    x=torch.from_numpy(runtime._aligned(rows[ids],True))
    batches=x.split(48)
    runtime.net.spatial_conv[1].k=None
    posterior=torch.cat([runtime.net(batch).softmax(1) for batch in batches])
    report={}
    for seed,(root,_) in ARCH.items():
        weights=Path(json.loads((root/'fit_lock.json').read_text())['root'])/f'J_{seed}.pt'
        joint.pin(weights,CHECKPOINTS[seed])
        net=init(runtime.net,'J')
        net.load_state_dict(torch.load(weights,weights_only=True,map_location='cpu'))
        layer=net.spatial_conv[1]
        items=torch.cat([layer.encode(layer.standardized(prior.preactivation(net,b))) for b in batches])
        signals=torch.stack([centers(items[sessions==s],y[sessions==s]) for s in selected])
        full=signals.mean(0)
        with np.load('prototypes.npz') as z:
            if not smoke:
                torch.testing.assert_close(full.float(),torch.from_numpy(z[str(seed)]),atol=5e-6,rtol=2e-5)
        records={}
        for fit in ('full','leave_session_out'):
            energies={kind:[] for kind in KINDS}
            projected={kind:[] for kind in KINDS}
            for i,s in enumerate(selected):
                use=sessions==s
                proto=full if fit=='full' else signals[torch.arange(len(selected))!=i].mean(0)
                writes,_=edit(items[use],posterior[use],proto)
                axis=proto/proto.norm()
                for kind,w in writes.items():
                    effect=centers(w,y[use])
                    energies[kind].append(float(effect.square().sum()))
                    projected[kind].append(float((effect*axis).sum().square()))
            records[fit]=dict(norm_squared={k:float(np.mean(v)) for k,v in energies.items()},
                              projection_squared={k:float(np.mean(v)) for k,v in projected.items()})
        report[seed]=records
    if not smoke:
        Path('diagnostic.json').write_text(json.dumps(dict(source_rows=ROWS,seeds=report),indent=2))


if __name__ == '__main__':
    main(sys.argv[1]=='smoke')
