import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch
from torch.nn import functional as F

from network import SetDecoder, subsets
from source_labels import read_training_labels
from train import DATA, PERM, ORDERS, digest, sha
from score379 import ORDER_SHA, DATA_SHA


def checks(net, features):
    q = features[:12]
    w = net.write(q)
    full = net.predict_session(q, w)
    total = torch.zeros_like(w[:1])
    decisions = []
    for i in range(len(q)):
        decisions.append(net.read(q[i:i+1], total, torch.full_like(total[:, :1], i)))
        total = total + w[i:i+1]
    assert torch.allclose(full, torch.cat(decisions), atol=2e-5)
    assert torch.equal(full, net.predict_session(q, w))
    assert torch.allclose(full[:5], net.predict_session(q[:5], w[:5]), atol=2e-5)
    assert torch.allclose(full[-1], net.predict_session(q, torch.cat((w[:-1].flip(0),w[-1:])))[-1], atol=2e-5)
    zero = net.predict_session(q, torch.zeros_like(w))
    assert torch.equal(zero[0], full[0]) and torch.isfinite(full).all()
    return dict(write_sd=float(w.std(0).mean()), zero_logit_rms=float((full-zero).square().mean().sqrt()))


def main(held, seed, smoke):
    assert 'A100' in torch.cuda.get_device_name() and sha(DATA)==DATA_SHA and sha(PERM)==ORDER_SHA
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    with np.load(DATA) as data:
        people = np.array([s.split('/')[-1] for s in data['subject']])
        streams = np.char.add(np.char.add(people, '|'), data['session'])
        interfaces = data['interface']
        x = torch.from_numpy(data['X']).float().cuda()
    fit, test = np.flatnonzero(people!=held), np.flatnonzero(people==held)
    assert len(set(people[fit]))==9 and len(test)
    x /= x[fit].square().mean().sqrt()
    labels = read_training_labels(DATA, fit, len(x))
    y = torch.full((len(x),), -1, device='cuda', dtype=torch.long)
    y[fit] = torch.from_numpy(labels).cuda()
    sessions = [fit[streams[fit]==s] for s in sorted(set(streams[fit]))]
    weight = torch.zeros(len(x), device='cuda')
    for rows in sessions:
        local = labels[np.searchsorted(fit, rows)]
        for iface in set(interfaces[rows]):
            for k in range(3):
                ids = rows[(interfaces[rows]==iface)&(local==k)]
                assert len(ids)
                weight[ids] = len(rows)/len(ids)/6
    rng = np.random.default_rng(seed)
    batches = [rng.permutation(sessions[rng.integers(len(sessions))]) for _ in range(2 if smoke else 360)]
    with np.load(PERM) as z:
        orders = {o:z[f'{o[:-1]}|{o[-1]}|all_loso'] for o in ORDERS}
    active = sessions[-1][:40] if smoke else test
    out, proof = {'index':active}, {}
    for arm in ('ce', 'consistent'):
        torch.manual_seed(seed)
        gen = torch.Generator(device='cuda').manual_seed(seed)
        net = SetDecoder().cuda()
        opt = torch.optim.AdamW(net.parameters(),lr=.001,weight_decay=.001)
        for rows in batches:
            opt.zero_grad()
            q = net.encode(x[rows])
            w = net.write(q)
            mask, count = subsets(len(rows), gen)
            logits = net.read(q.expand(2,-1,-1), mask @ w, count.expand(2,-1,-1))
            loss = (F.cross_entropy(logits.flatten(0,1),y[rows].repeat(2),reduction='none') *
                    weight[rows].repeat(2)).mean()
            centered = logits - logits.mean(-1,keepdim=True)
            penalty = (centered[0]-centered[1]).square().mean()
            loss = loss + (1. if arm=='consistent' else 0.) * penalty
            assert torch.isfinite(loss) and penalty > 0
            loss.backward()
            assert net.write[0].weight.grad.abs().max() > 0
            opt.step()
        net.eval()
        with torch.no_grad():
            feats = net.encode(x[active])
            proof[arm] = checks(net, feats)
            writes = net.write(feats)
            for order,p in orders.items():
                assert np.array_equal(np.sort(p),np.arange(len(x)))
                for s in sorted(set(streams[active])):
                    rows = p[np.isin(p,active[streams[active]==s])]
                    pos = np.searchsorted(active,rows)
                    donors = [d for d in sessions if people[d[0]] != people[rows[0]]]
                    d = max(donors,key=len)
                    donor_rows = p[np.isin(p,d)]
                    donor = net.write(net.encode(x[np.resize(donor_rows,len(rows))]))
                    for name,update in [('real',writes[pos]),('zero',torch.zeros_like(writes[pos])),('donor',donor)]:
                        logits = net.predict_session(feats[pos],update)
                        assert torch.isfinite(logits).all()
                        key = f'{arm}_{name}|{order}'
                        pred = out.setdefault(key,np.full(len(active),-1,dtype=np.int8))
                        pred[pos] = logits.argmax(1).cpu().numpy()
            assert all((p>=0).all() for p in out.values())
    if smoke:
        print('source-only matched two-arm smoke passed',proof,flush=True)
        return
    name = f'{held}_{seed}'
    np.savez_compressed(name+'.npz',**out)
    record = dict(held=held,seed=seed, source_people=sorted(set(people[fit])), train_rows=digest(fit),
                  outputs=sha(Path(name+'.npz')), source=sha(Path('source.sha256')), orders=ORDER_SHA,
                  episodes=hashlib.sha256(b''.join(b.tobytes() for b in batches)).hexdigest(),checks=proof)
    Path(name+'.json').write_text(json.dumps(record,indent=2))


if __name__=='__main__':
    main(sys.argv[1],int(sys.argv[2]),sys.argv[3:]==['smoke'])
