"""Restore the two six-person A6 fits and replay identical external rows."""
from pathlib import Path
import copy
import json
import sys
import numpy as np
import torch

BASE = Path('/mnt/lustre/work/gehler/gfh527')
PRIMARY = BASE / 'ni26_joint382_v1_20261118'
sys.path.insert(0, str(PRIMARY))
from restore import init, prior, joint
sys.path.insert(0, str(BASE/'ni26_joint382_actor_v1_20261118'))
from lock_fits import lock

ROOTS = {382: (PRIMARY, BASE/'ni26_joint382_actor_v1_20261118'),
         1382: (BASE/'ni26_joint382_h100_1382_20261118/fits', BASE/'ni26_joint382_h100_1382_20261118/actor')}


def models(native):
    result = {}
    for seed, (fit, actor) in ROOTS.items():
        assert lock(fit, seed) == json.loads((actor/'fit_lock.json').read_text())
        for arm in ('J', 'K'):
            model = init(native, arm)
            model.load_state_dict(torch.load(fit/f'{arm}_{seed}.pt', weights_only=True, map_location='cpu'))
            result[f'{arm}{seed}'] = model.eval()
            if arm == 'J':
                model.spatial_conv[1].mode = 'real'
                blank = copy.deepcopy(model)
                blank.spatial_conv[1].mode = 'blank'
                result[f'blank{seed}'] = blank
    for name, k in [('pop', None), ('native', 20.)]:
        model = copy.deepcopy(native).eval()
        model.spatial_conv[1].k = k
        result[name] = model
    return result


@torch.no_grad()
def run(person):
    _, _, source_rows, runtime = prior.source()
    prior.owners(source_rows)
    assert runtime.align is False
    nets = models(runtime.net)
    primary415 = BASE/'ni26_select415_20261118'
    shard = 6 + (person-1)//5
    bank = json.loads((primary415/f'cap_{shard}.json').read_text())
    records = [r for r in bank if r['kind']=='W' and r['person']==person]
    assert len(records)==2
    out = {}
    with np.load(BASE/'neural-interfaces-2026/data/shin2018/windows.npz') as z:
        rows = np.flatnonzero((z['subject'] == f'Shin2018/{person}') & (z['win']==0))
        assert len(rows)==60 and len(set(z['session'][rows]))==1
        raw = z['X'][rows]
    for record in records:
        seed = record['seed']
        order = np.random.default_rng([415, person, seed, 0]).permutation(60)
        assert np.array_equal(rows[order], record['raw'])
        x = torch.roll(torch.from_numpy(runtime._aligned(order, True, src=raw)), 0, -1)
        for arm, net in nets.items():
            full = net(x)
            pieces = torch.cat([net(block) for block in x.split(20)])
            torch.testing.assert_close(full[:20], pieces[:20], atol=3e-4, rtol=1e-5)
            assert torch.isfinite(full).all() and torch.isfinite(pieces).all()
            a, b = full.argmax(1).numpy(), pieces.argmax(1).numpy()
            assert np.array_equal(a[:20], b[:20])
            if arm.startswith('blank') or arm=='pop':
                assert np.array_equal(a,b)
            out[f'{seed}|{arm}'] = np.stack([a,b]).astype(np.int8)
        out[str(seed)] = rows[order]
    np.savez_compressed(f'person{person}.npz', **out)


if __name__=='__main__':
    run(int(sys.argv[1]))
