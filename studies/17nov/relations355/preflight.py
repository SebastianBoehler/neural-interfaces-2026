"""Real-source CPU shape/activity and independent delivery-order preflight; no fit/score."""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
from model import ARMS, Relations

root, ws = map(Path,sys.argv[1:])
sys.path.insert(0,str(ws/'evaluator'))
from orders352 import source_orders

torch.set_num_threads(1)
with np.load(root/'data/proteus_windows.npz') as z:
    d = {k:z[k].copy() for k in ('subject','session','task','run')}
    ids = np.flatnonzero(d['subject'] == 'Dreyer2026Proteus/P02')[:4]
    x = torch.from_numpy(z['X'][ids].copy())
    assert x.shape == (4,41,480) and torch.isfinite(x).all()
    assert len(set(z['ch_names'])) == 41
    d['stream'] = np.char.add(np.char.add(d['subject'],'|'),d['session'])
    d['group'] = np.char.add(np.char.add(d['task'],'|'),d['run'].astype(str))
orders = source_orders(root,d)
counts, states = {}, []
for arm in ('instant','lag6','offset','aligned'):
    torch.manual_seed(355)
    net = Relations(arm)
    states.append({k:v.clone() for k,v in net.state_dict().items()})
    counts[arm] = sum(v.numel() for v in net.parameters())
    value = net(x)
    value.square().mean().backward()
    assert net.filters.weight.grad.norm() > 0 and torch.isfinite(value).all()
assert len(set(counts.values())) == 1
assert all(all(torch.equal(v,states[0][k]) for k,v in state.items()) for state in states)
proof = dict(real_source='P02', rows=ids.tolist(), parameters=counts,
             all_initial_tensors_equal=True, all_four_real_source_gradients_active=True,
             eight_pinned_independent_orders={k:hashlib.sha256(v.tobytes()).hexdigest() for k,v in orders.items()},
             fit_or_accuracy=False)
(ws/'preflight.json').write_text(json.dumps(proof,indent=2)+'\n')
print(json.dumps(proof),flush=True)
