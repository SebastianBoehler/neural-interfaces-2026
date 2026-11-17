"""Reuse the pinned §379 fold runner with three matched feature extractors."""
import hashlib
import inspect
from pathlib import Path
import sys

import torch

OWNER = Path('/mnt/lustre/work/gehler/gfh527/ni26_wave379_v1_20261118')
sys.path.insert(0, str(OWNER))
import train
from decoder import ARMS, DualScale


def verify():
    assert train.sha(OWNER/'source.sha256') == 'a2fb59c104a29bf248ec843645c9fd4adbdbd0f3ee66b36e37ca2421374119c2'
    for line in (OWNER/'source.sha256').read_text().splitlines():
        h,name = line.split(maxsplit=1)
        assert train.sha(OWNER/name) == h
    assert Path(train.__file__) == OWNER/'train.py'


def initialize(arm):
    net = DualScale(arm).cuda()
    state = hashlib.sha256()
    for name, v in net.named_parameters():
        state.update(name.encode())
        state.update(v.detach().cpu().numpy().tobytes())
    h = state.hexdigest()
    if arm == ARMS[0]:
        initialize.first = h
    assert h == initialize.first, 'parameter count, shapes or initialization changed'
    return net


def check_gradients(net):
    for w in (*net.taps,net.mixer.weight,net.temporal.weight,net.spatial.weight):
        assert w.grad is not None and torch.isfinite(w.grad).all() and w.grad.abs().amax() > 0
    assert net.spatial.weight.grad.chunk(2)[1].abs().amax() > 0
    perm = net.permutation.reshape(4,28)
    assert torch.equal(perm.sort(1).values.flatten(), torch.arange(112,device=perm.device))
    assert (perm.flatten()!=torch.arange(112,device=perm.device)).sum() > 90


def guard_permutation(net, x):
    if net.arm != 'perm':
        return
    unpermuted = DualScale('spec').to(x.device).eval()
    unpermuted.load_state_dict(net.state_dict())
    a, b = net.encode(x), unpermuted.encode(x)
    assert torch.equal(a[:,:24], b[:,:24])
    assert (a[:,24:]-b[:,24:]).abs().max() > 1e-5


def main(person, smoke):
    verify()
    source = inspect.getsource(train.main)
    changes = {
        "for arm in ('static','norm','memory'):": 'for arm in ARMS:',
        'net = Decoder().cuda()': 'net = initialize(arm)',
        'for rows in batches:': 'for step,rows in enumerate(batches):',
        "if arm == 'memory':\n                assert net.write[0].weight.grad.abs().max() > 0": 'if step == 0:\n                check_gradients(net)',
        'features = net.encode(sample)': 'guard_permutation(net, sample)\n            features = net.encode(sample)',
        'donor = torch.cat(net.unpack(net.encode(x[sessions[0]])),-1)': 'donor = None',
    }
    for before, after in changes.items():
        if source.count(before) != 1:
            raise ValueError('unrecognized §379 runner: '+before)
        source = source.replace(before, after)
    scope = dict(train.main.__globals__, ARMS=ARMS, initialize=initialize, check_gradients=check_gradients,
                 guard_permutation=guard_permutation)
    exec(compile(source,'<pinned train379 with scale380 encoder>','exec'),scope)
    scope['main'](person,smoke)


if __name__ == '__main__':
    main(sys.argv[1],sys.argv[2:]==['smoke'])
