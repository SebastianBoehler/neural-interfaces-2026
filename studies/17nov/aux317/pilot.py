"""Bounded source-only §317 pilot. Outer P01 labels are never indexed."""
import copy
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

ROOT = Path(os.environ['NI26_ROOT'])
sys.path.insert(0, str(ROOT / 'experiments'))
import protocol
from encoder import Encoder, fingerprint
from train import auxiliary, train


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1048576), b''):
            h.update(b)
    return h.hexdigest()


def identity_guard(x, y, sx, sy):
    """Separate main step then discarded auxiliary step equals main-only update exactly."""
    net = Encoder('fb', 0, 'cpu')
    twin = copy.deepcopy(net)
    head = nn.Linear(net.dim, 2)
    opts = [torch.optim.AdamW(list(n.body.parameters()) + list(n.task.parameters()), lr=.001)
            for n in (net, twin)]
    # Dropout RNG state is identical for both independent main paths.
    rng = torch.get_rng_state()
    for n, opt in zip((net, twin), opts):
        torch.set_rng_state(rng); n.train(); opt.zero_grad(set_to_none=True)
        nn.functional.cross_entropy(n(x), y).backward(); opt.step()
    if fingerprint(net) != fingerprint(twin):
        raise ValueError('PROTEUS control update mismatch before auxiliary step')
    before = fingerprint(net)
    aux = torch.optim.AdamW(list(net.body.parameters()) + list(head.parameters()), lr=.001)
    opts[0].zero_grad(set_to_none=True); aux.zero_grad(set_to_none=True)
    auxiliary(net, head, sx, sy, False); aux.step()
    if fingerprint(net) != before or fingerprint(net) != fingerprint(twin):
        raise ValueError('stopped auxiliary total encoder update differs from main-only twin')
    return dict(main_only_total_update_bitwise=True, aux_bn_buffers_bitwise=True,
                auxiliary_mode_restored=True, auxiliary_head_gradient_nonzero=True)


def main():
    if os.environ['NI26_DEVICE'] != 'cpu':
        raise ValueError('this registered route is CPU only')
    torch.set_num_threads(6)
    ws = Path.cwd()
    epochs = int(os.environ.get('NI26_EPOCHS', '40'))
    if epochs not in (1, 40):
        raise ValueError('registered smoke1 or complete40 epochs only')
    d = protocol.load()
    people = sorted(set(d['subject'].tolist()))
    held = next(p for p in people if p.endswith('/P01'))
    val = next(p for p in people if p.endswith('/P02'))
    fit = [p for p in people if p not in (held, val)]
    rows = np.flatnonzero(np.isin(d['subject'], fit))
    vr = np.flatnonzero(d['subject'] == val)
    if len(fit) != 8 or held in set(d['subject'][rows]) or val in set(d['subject'][rows]):
        raise ValueError('PROTEUS source exclusion failed')
    # Only source-fit and source-validation labels are materialized below.
    x = torch.from_numpy(d['X'][rows].astype(np.float32))
    vx = torch.from_numpy(d['X'][vr].astype(np.float32))
    y, vy = torch.from_numpy(d['y'][rows]).long(), torch.from_numpy(d['y'][vr]).long()
    scale = float(x.square().mean().sqrt()); x /= scale; vx /= scale
    path = ROOT / 'data/shin2018/windows.npz'
    with np.load(path, allow_pickle=False) as z:
        if z['X'].shape != (3120, 41, 480) or z['classes'].tolist() != ['baseline', 'WG']:
            raise ValueError('wrong full Shin export or class order')
        if not np.array_equal(z['ch_names'], d['ch_names']) or len(z['recorded']) != 16:
            raise ValueError('Shin channel provenance mismatch')
        if len(np.unique(z['subject'])) != 26 or np.bincount(z['y']).tolist() != [1560,1560]:
            raise ValueError('wrong Shin cohort/counts')
        people_s = np.array([int(p.split('/')[-1]) for p in z['subject']])
        sr = np.flatnonzero((people_s > 5) & (z['win'] == 0))
        svr = np.flatnonzero((people_s <= 5) & (z['win'] == 0))
        sx = torch.from_numpy(z['X'][sr].astype(np.float32))
        svx = torch.from_numpy(z['X'][svr].astype(np.float32))
        sy, svy = torch.from_numpy(z['y'][sr]).long(), torch.from_numpy(z['y'][svr]).long()
        ss = z['subject'][sr]
        keys = [(str(z['subject'][r]), str(z['block'][r]), int(z['trial'][r])) for r in np.r_[sr,svr]]
        if len(set(keys)) != len(keys):
            raise ValueError('duplicate trial contributes multiple crops')
        channels = dict(recorded=z['recorded'].tolist(), interpolated=z['interpolated'].tolist())
    sscale = float(sx.square().mean().sqrt()); sx /= sscale; svx /= sscale
    if not all(np.isfinite(s) and s > 0 for s in (scale,sscale)):
        raise ValueError('invalid source-only scale')
    report = dict(status='running_source_only', fit_people=fit, inner_validation=val, outer_excluded=held,
                  epochs=epochs, source_sha256=sha(ws/'source.sha256'),
                  proteus_sha256=sha(ROOT/'data/proteus_windows.npz'), shin_sha256=sha(path),
                  source_rows_sha256=hashlib.sha256(rows.tobytes()).hexdigest(),
                  auxiliary_rows_sha256=hashlib.sha256(sr.tobytes()).hexdigest(), channels=channels,
                  scales=dict(proteus=scale,shin=sscale),
                  guard=identity_guard(x[:8],y[:8],sx[:8],sy[:8]), arms=[])
    for arm in ('A','B','C'):
        report['arms'].append(train(arm,x,y,vx,vy,sx,sy,ss,svx,svy,epochs))
        (ws/'progress.json').write_text(json.dumps(report,indent=2))
    if len({a['source_plan'] for a in report['arms']}) != 1:
        raise ValueError('main source minibatch exposure differs between arms')
    if len({a['initial'] for a in report['arms']}) != 1:
        raise ValueError('initial main encoders differ')
    report['status']='source_only_pilot_complete'
    (ws/'result.json').write_text(json.dumps(report,indent=2))


if __name__ == '__main__':
    main()
