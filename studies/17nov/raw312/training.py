"""Source-exclusive raw sampling, fold normalization and task training."""
import json
from pathlib import Path

import numpy as np
import torch
from encoder import Encoder, fingerprint, pretrain, supervised
from data import sha


def sources(d, held, seed):
    universe = sorted(set(d['subject'].tolist()))
    target = next(p for p in universe if p.split('/')[-1] == held)
    fit = [p for p in universe if p != target]
    val = fit[seed]
    if len(fit) != 9:
        raise ValueError('expected exactly nine source people')
    return target, fit, val


def tensors(d, people, device):
    rows = np.flatnonzero(np.isin(d['subject'], people))
    return rows, torch.as_tensor(d['X'][rows], device=device), torch.as_tensor(d['y'][rows], device=device).long()


def raw_rows(cache, people, seed):
    rng = np.random.default_rng(seed + 312)
    values, hashes = [], {}
    for person in people:
        path = Path(cache) / f'{person.split("/")[-1]}.npz'
        hashes[str(path)] = sha(path)
        with np.load(path, allow_pickle=False) as z:
            if str(z['person']) != person.split('/')[-1]:
                raise ValueError('raw source person mismatch')
            x, record = z['X'], z['record']
            records = sorted(set(record.tolist()))
            # Exactly64 per person; each draw chooses recording then crop uniformly.
            chosen = [rng.choice(np.flatnonzero(record == records[rng.integers(len(records))])) for _ in range(64)]
            values.append(x[chosen])
    return np.concatenate(values), hashes


def fit(d, cache, people, arch, arm, seed, device, epochs=40, val_people=None, pilot=False):
    rows, x, y = tensors(d, people, device)
    # One scalar fitted on source rows, no per-window centering/RMS.
    scale = float(x.square().mean().sqrt())
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError('invalid source scale')
    x = x / scale
    net = Encoder(arch, seed, device)
    initial = fingerprint(net.body)
    raw, hashes = raw_rows(cache, people, seed)
    raw = torch.as_tensor(raw, device=device) / scale
    pre_epochs = 1 if pilot else 20
    pre_steps = pre_epochs * ((len(raw) + 63) // 64)
    pre = None
    if arm in ('F', 'T'):
        pre = pretrain(net, raw, seed, epochs=pre_epochs)
        if fingerprint(net.body) == initial:
            raise ValueError('pretraining left encoder unchanged')
    if arm == 'F':
        for parameter in net.body.parameters():
            parameter.requires_grad_(False)
        net.frozen = True
    validation = None
    if val_people:
        _, vx, vy = tensors(d, val_people, device)
        validation = (vx / scale, vy)
    task = supervised(net, x, y, seed, epochs, extra_steps=pre_steps if arm == 'C' else 0, val=validation)
    net.eval()
    return net, scale, dict(fit_people=people, raw_hashes=hashes, scale=scale,
        initial_encoder=initial, final_encoder=fingerprint(net.body), pretraining=pre,
        moment_scaler_sha256=__import__('hashlib').sha256(torch.stack([net.moment_mean, net.moment_sd]).detach().cpu().numpy().tobytes()).hexdigest(),
        supervised=task, normalization='source-only scalar;mean/power skip',
        source_rows_sha256=__import__('hashlib').sha256(rows.tobytes()).hexdigest())


def probabilities(net, x, scale, device):
    net.eval()
    with torch.no_grad():
        p = torch.cat([net(torch.as_tensor(b, device=device) / scale).double().softmax(1).cpu()
                       for b in np.array_split(x, max(1, (len(x) + 31) // 32))]).numpy()
    if not np.isfinite(p).all() or p.shape != (len(x), 3) or not np.allclose(p.sum(1), 1):
        raise ValueError('invalid probabilities')
    return p
