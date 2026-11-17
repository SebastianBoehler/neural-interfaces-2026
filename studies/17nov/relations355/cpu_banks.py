"""One fixed arm/process: complete CPU B1 banks without labels or refitting."""
import copy
import json
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np
import torch

old, ws = map(Path,(os.environ['R355_WS'],os.environ['R355_CPU']))
arm = os.environ['R355_ARM']
os.sched_setaffinity(0,{int(os.environ['R355_CORE'])})
torch.set_num_threads(1); torch.set_num_interop_threads(1)
sys.path[:0] = [str(old/'code'),str(old/'evaluator')]
from model import ARMS, Relations
from run import sha
from orders352 import source_orders

assert torch.get_default_dtype() == torch.float32
family = next(f for f,a in ARMS.items() if arm in a)
root = Path('/mnt/lustre/work/gehler/gfh527/neural-interfaces-2026')
t = time.monotonic()
with np.load(root/'data/proteus_windows.npz') as z:
    x = torch.from_numpy(z['X'].copy())
    d = {k:z[k].copy() for k in ('subject','session','task','run')}
assert x.shape == (2655,41,480) and x.dtype == torch.float32 and torch.isfinite(x).all()
d['stream'] = np.char.add(np.char.add(d['subject'],'|'),d['session'])
d['group'] = np.char.add(np.char.add(d['task'],'|'),d['run'].astype(str))
orders = source_orders(root,d)
original = json.loads((old/'output'/family/'proof.json').read_text())
assert len(original['folds']) == 10 and original['epochs'] == 40
out = ws/'partitions'/arm
out.mkdir(parents=True,exist_ok=False)
proof = dict(arm=arm,family=family,device='cpu',dtype='float32',torch=torch.__version__,
    threads=torch.get_num_threads(),interop_threads=torch.get_num_interop_threads(),
    mkldnn=torch.backends.mkldnn.enabled,platform=platform.platform(),
    affinity=sorted(os.sched_getaffinity(0)),labels_opened=False,folds=[],
    fit_proof_sha256=sha(old/'output'/family/'proof.json'))


def load(checkpoint):
    net = Relations(arm)
    state = torch.load(checkpoint,map_location='cpu',weights_only=True)
    assert all(not v.is_floating_point() or v.dtype == torch.float32 for v in state.values())
    net.load_state_dict(state)
    return net.eval()


@torch.no_grad()
def b1(net,rows):
    return np.stack([net(x[i:i+1]).softmax(1)[0].numpy() for i in rows])


with torch.no_grad():
    for rec in original['folds']:
        held = rec['held']; rows = np.flatnonzero(d['subject'] == held)
        assert rec['source_people'] == sorted(set(d['subject'])-{held})
        name = f'{held.split("/")[-1]}_{arm}'
        checkpoint = old/'output'/family/f'{name}.pt'
        assert sha(checkpoint) == rec['arms'][arm]['checkpoint_sha256']
        net = load(checkpoint)
        state = {k:v.clone() for k,v in net.state_dict().items()}
        reference = None
        for order, permutation in orders.items():
            sequence = permutation[np.isin(permutation,rows)]
            current = b1(net,sequence)[np.argsort(sequence)]
            if reference is None:
                reference = current
            else:
                assert np.array_equal(reference,current), (held,arm,order)
        short = rows[:17]
        assert np.array_equal(b1(load(checkpoint),short),reference[:17])
        assert np.array_equal(b1(copy.deepcopy(net),short),reference[:17])
        b1(net,np.flatnonzero(d['subject'] != held)[:7])
        assert np.array_equal(b1(net,short),reference[:17])
        assert all(torch.equal(v,net.state_dict()[k]) for k,v in state.items())
        with np.load(old/'output'/family/f'{name}.npz') as z:
            assert np.array_equal(z['rows'],rows)
            previous = z['probability'].copy()
        assert reference.shape == (len(rows),3) and np.isfinite(reference).all()
        assert np.all(reference >= 0) and np.allclose(reference.sum(1),1,atol=1e-6)
        np.savez(out/f'{name}.npz',rows=rows,probability=reference)
        delta = reference-previous
        record = dict(held=held,arm=arm,rows=len(rows),bank_sha256=sha(out/f'{name}.npz'),
            eight_actual_orders_bitwise=True,reset_copy_prefix_interleaving_exact=True,
            unchanged_state=True,checkpoint_sha256=sha(checkpoint),
            b64_maxabs=float(np.abs(delta).max()),
            b64_relative_l2=float(np.linalg.norm(delta)/max(np.linalg.norm(previous),1e-30)),
            b64_integer_changes=int(np.sum(reference.argmax(1)!=previous.argmax(1))))
        proof['folds'].append(record)
        (out/'progress.json').write_text(json.dumps(proof,indent=2))
        print(json.dumps(record),flush=True)
proof['seconds'] = time.monotonic()-t
(out/'proof.json').write_text(json.dumps(proof,indent=2)+'\n')
