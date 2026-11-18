"""Repeat the fitted model with a fresh initialization and isolate write content from age."""
import json
from pathlib import Path
import sys

import numpy as np
import torch
import fit
from model import Decoder
import train


@torch.no_grad()
def controls(person):
    checkpoint = Path(person+'_memory.pt')
    net = Decoder().cuda().eval()
    net.load_state_dict(torch.load(checkpoint, weights_only=True))
    with np.load(train.DATA) as z:
        people = np.array([s.split('/')[-1] for s in z['subject']])
        streams = np.char.add(np.char.add(people,'|'), z['session'])
        x = torch.from_numpy(z['X']).float().cuda()
    test = np.flatnonzero(people == person)
    src = np.flatnonzero(people != person)
    x /= x[src].square().mean().sqrt()
    def encoded(rows):
        return torch.cat([net.encode(x[b]) for b in np.array_split(rows, 1+len(rows)//32)])
    means = []
    for session in np.unique(streams[src]):
        rows = src[streams[src]==session]
        means.append(torch.cat(net.unpack(encoded(rows)), -1).mean(0))
    constant = torch.stack(means).mean(0)
    assert torch.isfinite(constant).all() and constant[-48:].abs().max() > 1e-5
    real = encoded(test)
    out = {arm:np.full((8,len(test)), -1, dtype=np.int8) for arm in ('zero_write','source_write')}
    with np.load(train.PERM) as orders, np.load(person+'.npz') as original:
        for i,o in enumerate(train.ORDERS):
            order = orders[f'{o[:-1]}|{o[-1]}|all_loso']
            for session in np.unique(streams[test]):
                rows = order[streams[order] == session]
                pos = np.searchsorted(test,rows)
                assert np.array_equal(test[pos],rows)
                logits = net.sequence(real[pos],'memory')[0]
                assert np.array_equal(logits.argmax(1).cpu().numpy(), original['memory|'+o][pos])
                for arm in out:
                    writes = constant.expand(len(pos),-1)
                    if arm == 'zero_write':
                        writes = torch.zeros_like(writes)
                    ctrl = net.sequence(real[pos],'memory',donor=writes)[0]
                    assert torch.equal(ctrl[0],logits[0]) and torch.isfinite(ctrl).all()
                    out[arm][i,pos] = ctrl.argmax(1).cpu().numpy()
    assert all(np.isin(v,(0,1,2)).all() for v in out.values())
    dest = Path(person+'.controls.npz')
    np.savez_compressed(dest,index=test,**out)
    Path(person+'.controls.json').write_text(json.dumps(dict(sha=train.sha(dest),checkpoint=train.sha(checkpoint))))


if __name__ == '__main__':
    person, arm = sys.argv[1:3]
    smoke = sys.argv[3:] == ['smoke']
    fit.run(person, arm, smoke, init_seed=1441)
    if not smoke:
        controls(person)
