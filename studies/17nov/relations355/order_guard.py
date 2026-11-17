"""Actual B1 eight-order frozen inference; no query labels or accuracy."""
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

ws = Path(os.environ['R355_WS'])
root = Path('/mnt/lustre/work/gehler/gfh527/neural-interfaces-2026')
sys.path[:0] = [str(ws/'code'), str(ws/'evaluator')]
from model import ARMS, Relations
from run import setup, sha
from orders352 import source_orders

setup()
started = time.monotonic()
with np.load(root/'data/proteus_windows.npz') as z:
    x = torch.from_numpy(z['X'].copy()).cuda()
    d = {k:z[k].copy() for k in ('subject','session','task','run')}
d['stream'] = np.char.add(np.char.add(d['subject'],'|'),d['session'])
d['group'] = np.char.add(np.char.add(d['task'],'|'),d['run'].astype(str))
orders = source_orders(root,d)
proof = dict(labels_opened=False, folds=[], eight_orders_actual_B1=True)
with torch.no_grad():
    for family, arms in ARMS.items():
        original = json.loads((ws/'output'/family/'proof.json').read_text())
        assert len(original['folds']) == 10 and original['epochs'] == 40
        for rec in original['folds']:
            held = rec['held']; rows = np.flatnonzero(d['subject'] == held)
            for arm in arms:
                checkpoint = ws/'output'/family/f'{held.split("/")[-1]}_{arm}.pt'
                assert sha(checkpoint) == rec['arms'][arm]['checkpoint_sha256']
                net = Relations(arm).cuda()
                net.load_state_dict(torch.load(checkpoint, weights_only=True))
                net.eval()
                state = {k:v.clone() for k,v in net.state_dict().items()}
                reference = None
                for name, permutation in orders.items():
                    sequence = permutation[np.isin(permutation, rows)]
                    values = np.empty((len(d['subject']),3), dtype='float32')
                    for i in sequence:
                        values[i] = net(x[i:i+1]).softmax(1)[0].cpu().numpy()
                    current = values[rows].copy()
                    if reference is None:
                        reference = current
                    else:
                        assert np.array_equal(reference,current), (held,arm,name,'actual order mismatch')
                with np.load(ws/'output'/family/f'{held.split("/")[-1]}_{arm}.npz') as z:
                    assert np.array_equal(z['rows'],rows)
                    base = z['probability'].copy()
                error = float(np.max(np.abs(reference-base)))
                assert error <= 1e-6 and np.array_equal(reference.argmax(1),base.argmax(1)), (held,arm,error)
                assert all(torch.equal(v,net.state_dict()[k]) for k,v in state.items())
                proof['folds'].append(dict(held=held,arm=arm,b1_b64_max=error,
                    eight_actual_orders_bitwise=True,unchanged_state=True,
                    b1_sha256=hashlib.sha256(reference.tobytes()).hexdigest()))
                print(json.dumps(proof['folds'][-1]),flush=True)
                (ws/'order_progress.json').write_text(json.dumps(proof,indent=2))
assert len(proof['folds']) == 40
proof['seconds'] = time.monotonic()-started
(ws/'order_guard.json').write_text(json.dumps(proof,indent=2)+'\n')
print('ALL40_FROZEN_MODELS_EIGHT_ACTUAL_ORDERS_PASSED_NO_LABELS',flush=True)
