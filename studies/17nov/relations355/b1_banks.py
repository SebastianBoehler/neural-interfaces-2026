"""Fresh official-B1 banks from unchanged fitted weights; never requests held labels."""
import copy
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

ws, old = map(Path,(os.environ['R355_B1'],os.environ['R355_WS']))
root = Path('/mnt/lustre/work/gehler/gfh527/neural-interfaces-2026')
sys.path[:0] = [str(old/'code'),str(old/'evaluator')]
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
proof = dict(labels_opened=False,folds=[],execution_mode='B1',orders=list(orders),
             source={p.name:sha(p) for p in (ws/'instrument').glob('*') if p.is_file()},
             torch=torch.__version__,device=torch.cuda.get_device_name(),tf32=False)


def load(arm,checkpoint):
    net = Relations(arm).cuda()
    net.load_state_dict(torch.load(checkpoint,weights_only=True))
    return net.eval()


@torch.no_grad()
def b1(net,rows):
    return np.stack([net(x[i:i+1]).softmax(1)[0].cpu().numpy() for i in rows])


with torch.no_grad():
    for family, arms in ARMS.items():
        original = json.loads((old/'output'/family/'proof.json').read_text())
        assert len(original['folds']) == 10 and original['epochs'] == 40
        target = ws/'output'/family
        target.mkdir(parents=True,exist_ok=False)
        updated = copy.deepcopy(original)
        updated['execution_mode'] = 'B1'
        updated['fit_proof_sha256'] = sha(old/'output'/family/'proof.json')
        for rec in updated['folds']:
            held = rec['held']; rows = np.flatnonzero(d['subject'] == held)
            assert rec['source_people'] == sorted(set(d['subject'])-{held})
            for arm in arms:
                name = f'{held.split("/")[-1]}_{arm}'
                checkpoint = old/'output'/family/f'{name}.pt'
                assert sha(checkpoint) == rec['arms'][arm]['checkpoint_sha256']
                net = load(arm,checkpoint)
                state = {k:v.clone() for k,v in net.state_dict().items()}
                if not proof['folds']:
                    source = np.flatnonzero(d['subject'] != held)[:64]
                    b1(net,source[:4])
                    torch.cuda.synchronize(); t = time.monotonic()
                    b1(net,source); b1(net,source)
                    torch.cuda.synchronize(); elapsed = time.monotonic()-t
                    projected = (8*4*2655+4*10*3*17)/128*elapsed*1.5+60
                    proof['source_smoke'] = dict(rows=source.tolist(),seconds=elapsed,
                                                projected_seconds=projected,excluded_person=held)
                    (ws/'smoke.json').write_text(json.dumps(proof['source_smoke'],indent=2))
                    assert projected <= 540, ('prospective cost stop',projected)
                reference = None
                for order, permutation in orders.items():
                    sequence = permutation[np.isin(permutation,rows)]
                    values = b1(net,sequence)
                    current = values[np.argsort(sequence)]
                    if reference is None:
                        reference = current
                    else:
                        assert np.array_equal(reference,current), (held,arm,order)
                short = rows[:17]
                fresh = load(arm,checkpoint)
                assert np.array_equal(b1(fresh,short),reference[:17])
                twin = copy.deepcopy(net)
                assert np.array_equal(b1(twin,short),reference[:17])
                b1(net,np.flatnonzero(d['subject'] != held)[:7])
                assert np.array_equal(b1(net,short),reference[:17])
                assert all(torch.equal(v,net.state_dict()[k]) for k,v in state.items())
                with np.load(old/'output'/family/f'{name}.npz') as z:
                    assert np.array_equal(z['rows'],rows)
                    previous = z['probability'].copy()
                delta = reference-previous
                record = dict(held=held,arm=arm,rows=len(rows),eight_actual_orders_bitwise=True,
                    reset_copy_prefix_interleaving_exact=True,unchanged_state=True,
                    b64_maxabs=float(np.abs(delta).max()),
                    b64_relative_l2=float(np.linalg.norm(delta)/max(np.linalg.norm(previous),1e-30)),
                    b64_integer_changes=int(np.sum(reference.argmax(1)!=previous.argmax(1))),
                    checkpoint_sha256=sha(checkpoint))
                np.savez(target/f'{name}.npz',rows=rows,probability=reference)
                (target/f'{name}.pt').symlink_to(checkpoint)
                rec['arms'][arm]['bank_sha256'] = sha(target/f'{name}.npz')
                rec['arms'][arm]['new_b1_guards'] = record
                proof['folds'].append(record)
                print(json.dumps(record),flush=True)
                (ws/'progress.json').write_text(json.dumps(proof,indent=2))
        (target/'proof.json').write_text(json.dumps(updated,indent=2)+'\n')
assert len(proof['folds']) == 40
proof['seconds'] = time.monotonic()-started
(ws/'b1_guard.json').write_text(json.dumps(proof,indent=2)+'\n')
print('ALL40_B1_BANKS_AND_EIGHT_ACTUAL_ORDERS_QUALIFIED_NO_LABELS',flush=True)
