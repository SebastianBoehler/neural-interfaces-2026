import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
from torch.nn import functional as F

from model import Decoder
from source_labels import read_training_labels

BASE = Path('/mnt/lustre/work/gehler/gfh527/ni26_s318_coral_fbl_20261117')
DATA = BASE/'data/proteus_windows.npz'
PERM = BASE/'runs/shuffle_perms.npz'
ORDERS = ('within0','within1','within2','within3','full0','full1','runs0','runs1')


def sha(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle,'sha256').hexdigest()


def digest(x):
    return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()


def main(held, smoke):
    start = time.monotonic()
    assert 'A100' in torch.cuda.get_device_name()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    assert sha(DATA) == '2be4187fe33a5d476145cbb6d46164093b9b5c337209482503761042343e1327'
    order_hash = sha(PERM)
    assert order_hash == '6046bf5b2231d29d28bbfcf88ea7665d4aaa506c5787bd836ebec6a5b945430f'
    with np.load(DATA) as z:
        people = np.array([s.split('/')[-1] for s in z['subject']])
        streams = np.char.add(np.char.add(people,'|'),z['session'])
        interfaces = z['interface']
        x = torch.from_numpy(z['X']).float().cuda()
    fit = np.flatnonzero(people != held)
    test = np.flatnonzero(people == held)
    source = sorted(set(people[fit]))
    assert len(source) == 9 and len(test) and held not in source and x.shape == (2655,41,480)
    x /= x[fit].square().mean().sqrt()
    train_y = read_training_labels(DATA,fit,len(x))
    y = torch.full((len(x),),-1,device='cuda',dtype=torch.long)
    y[fit] = torch.from_numpy(train_y).cuda()
    sessions = [fit[streams[fit]==s] for s in sorted(set(streams[fit]))]
    weight = torch.zeros(len(x),device='cuda')
    for rows in sessions:
        local_y = train_y[np.searchsorted(fit,rows)]
        for interface in set(interfaces[rows]):
            for task in range(3):
                selected = rows[(interfaces[rows]==interface)&(local_y==task)]
                assert len(selected)
                weight[selected] = len(rows)/len(selected)/6
    rng = np.random.default_rng(379)
    batches = [rng.choice(sessions[rng.integers(len(sessions))],size=96,replace=False)
               for _ in range(2 if smoke else 360)]
    active = fit[:12] if smoke else test
    output = {'index':active}
    proofs = {}
    with np.load(PERM) as z:
        orders = {o:z[f'{o[:-1]}|{o[-1]}|all_loso'] for o in ORDERS}
    for perm in orders.values():
        assert np.array_equal(np.sort(perm),np.arange(len(x)))
    for arm in ('static','norm','memory'):
        torch.manual_seed(379)
        net = Decoder().cuda()
        opt = torch.optim.AdamW(net.parameters(),lr=.001,weight_decay=.001)
        for rows in batches:
            opt.zero_grad()
            logits,_ = net.step(x[rows],arm)
            loss = (F.cross_entropy(logits,y[rows],reduction='none')*weight[rows]).mean()
            loss.backward()
            assert torch.isfinite(loss)
            if arm == 'memory':
                assert net.write[0].weight.grad.abs().max() > 0
            opt.step()
        net.eval()
        with torch.no_grad():
            sample = x[active[:12]]
            features = net.encode(sample)
            full,_ = net.sequence(features,arm)
            repeated, state = [], None
            for row in sample:
                logit,state = net.step(row[None],arm,state)
                repeated.append(logit)
            assert torch.allclose(full,torch.cat(repeated),rtol=1e-4,atol=2e-4)
            assert torch.allclose(full[:5],net.sequence(features[:5],arm)[0],atol=2e-5)
            assert torch.equal(full,net.sequence(features,arm)[0])
            if arm == 'memory':
                blank = net.sequence(features,'static')[0]
                assert (full[1:]-blank[1:]).abs().max() > 1e-5
            if smoke:
                continue
            torch.save(net.state_dict(),f'{held}_{arm}.pt')
            proofs[arm] = sha(Path(f'{held}_{arm}.pt'))
            enc = torch.cat([net.encode(x[r]) for r in np.array_split(test,1+len(test)//32)])
            donor = torch.cat(net.unpack(net.encode(x[sessions[0]])),-1)
            names = (arm,'blank','donor') if arm=='memory' else (arm,)
            for order,p in orders.items():
                for name in names:
                    pred = np.full(len(test),-1)
                    for s in set(streams[test]):
                        rows = p[np.isin(p,test[streams[test]==s])]
                        pos = np.searchsorted(test,rows)
                        other = donor[torch.arange(len(pos),device='cuda')%len(donor)] if name=='donor' else None
                        logits,_ = net.sequence(enc[pos],'static' if name=='blank' else arm,donor=other)
                        assert torch.isfinite(logits).all()
                        pred[pos] = logits.argmax(1).cpu().numpy()
                    assert (pred>=0).all()
                    output[name+'|'+order] = pred
    if smoke:
        print('source-only smoke passed',flush=True)
        return
    path = Path(held+'.npz')
    np.savez_compressed(path,**output)
    record = dict(held=held,train_people=source,train_rows_sha256=digest(fit),order_sha256=order_hash,
                  outputs_sha256=sha(path),source_sha256=sha(Path('source.sha256')),seconds=time.monotonic()-start,
                  guard=dict(prefix=True,reset=True,step_equivalence=True,active=True,finite=True),
                  checkpoints=proofs, source_episodes=digest(np.stack(batches)),device=torch.cuda.get_device_name())
    Path(held+'.json').write_text(json.dumps(record,indent=2))
    print(held,record['seconds'],flush=True)


if __name__ == '__main__':
    main(sys.argv[1],sys.argv[2:] == ['smoke'])
