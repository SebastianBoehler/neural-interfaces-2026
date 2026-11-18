"""Fit a subject-generalizing multiple-kernel classifier on official EEG windows."""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from score379 import BASE, DATA_SHA, ORDER_SHA, ORDERS, PEOPLE, sha
from source_labels import read_training_labels


def distances(x):
    bands = ((4, 8), (8, 12), (12, 18), (18, 25), (25, 36))
    x = x.double()
    x = x - x.mean(1, keepdim=True)
    x = x - x.mean(-1, keepdim=True)
    freq = torch.fft.rfftfreq(480, 1 / 120, device=x.device)
    spec = torch.fft.rfft(x * torch.hann_window(480, periodic=False, device=x.device))
    tri = torch.triu_indices(41, 41, 1, device=x.device)
    blocks = []
    for lo, hi in bands:
        band = spec[:, :, (freq >= lo) & (freq < hi)]
        cov = (band @ band.mH).real
        power = cov.diagonal(dim1=1, dim2=2).clamp_min(1e-15)
        corr = cov * power.rsqrt()[:, :, None] * power.rsqrt()[:, None, :]
        for features in (power.log(), corr[:, tri[0], tri[1]]):
            d = torch.cdist(features.contiguous(), features.contiguous()).square()
            d.fill_diagonal_(0)
            blocks.append(d)
    return torch.stack(blocks)


def kernels(d, s, q):
    train = d[:, s[:, None], s]
    a, b = torch.triu_indices(len(s), len(s), 1, device=d.device)
    scale = train[:, a, b].median(1).values[:, None, None]
    assert (scale > 0).all()
    return (-train / scale).exp(), (-d[:, q[:, None], s] / scale).exp()


def classify(k, q, mix, penalty, y):
    kernel = (k * mix[:, None, None]).sum(0)
    dual = torch.linalg.solve(kernel + penalty.diag(), y)
    return 4 * ((q * mix[:, None, None]).sum(0) @ dual)


def main(held, seed, smoke):
    start = time.monotonic()
    assert held in PEOPLE and torch.cuda.is_available()
    path = BASE / 'data/proteus_windows.npz'
    assert sha(path) == DATA_SHA and sha(BASE/'runs/shuffle_perms.npz') == ORDER_SHA
    with np.load(path) as z:
        people = np.array([s.split('/')[-1] for s in z['subject']])
        fit = np.flatnonzero(people != held)
        test = np.flatnonzero(people == held)
        index = fit if smoke else np.r_[fit, test]
        cells = np.char.add(z['session'][fit], z['interface'][fit])
        x = torch.from_numpy(z['X'][index]).cuda()
    labels = read_training_labels(path, fit, len(people))
    people_fit = people[fit]
    groups = {(p, c): np.flatnonzero((people_fit == p) & (labels == c))
              for p in sorted(set(people_fit)) for c in range(3)}
    d = distances(x)
    y = F.one_hot(torch.from_numpy(labels.astype(np.int64)).cuda(), 3).double()
    rng = np.random.default_rng(seed)
    params = torch.zeros(10, dtype=torch.double, device='cuda', requires_grad=True)
    uniform = torch.full_like(params, .1)
    assert torch.equal(params.softmax(0), uniform)
    opt = torch.optim.Adam([params], lr=.04)
    for step in range(2 if smoke else 160):
        pquery = rng.choice(sorted(set(people_fit)))
        support = np.concatenate([rng.choice(row, 8, replace=False)
                                  for (p, _), row in groups.items() if p != pquery])
        query = np.concatenate([rng.choice(groups[pquery, c], 6, replace=False) for c in range(3)])
        assert not np.any(people_fit[support] == pquery)
        s, q = [torch.as_tensor(v, device='cuda') for v in (support, query)]
        k, kq = kernels(d, s, q)
        penalty = torch.ones(len(s), dtype=torch.double, device='cuda')
        logits = classify(k, kq, params.softmax(0), penalty, y[s])
        if step == 0:
            assert torch.equal(logits, classify(k, kq, uniform, penalty, y[s]))
        loss = F.cross_entropy(logits, y[q])
        opt.zero_grad()
        loss.backward()
        assert params.grad.abs().max() > 0 and params.grad.isfinite().all()
        opt.step()
    learned = params.detach().softmax(0)
    donor = learned.reshape(5, 2).roll(2, 0).flatten()
    assert (learned - donor).abs().max() > 1e-5
    if smoke:
        print('source-only CUDA forward/solve/backward passed', flush=True)
        return
    s = torch.arange(len(fit), device='cuda')
    q = torch.arange(len(fit), len(index), device='cuda')
    k, kq = kernels(d, s, q)
    key = np.array([f'{p}|{cell}|{label}' for p, cell, label in zip(people_fit, cells, labels)])
    _, inv, count = np.unique(key, return_inverse=True, return_counts=True)
    w = 1 / count[inv]
    penalty = torch.from_numpy(w.mean()/w).cuda()
    output = {'index': test}
    for arm, mix in [('U', uniform), ('L', learned), ('D', donor)]:
        pred = classify(k, kq, mix, penalty, y)
        assert pred.isfinite().all()
        # Perform an independent prefix query, including distance extraction and ridge solve.
        _, first = kernels(d, s, q[:7])
        assert torch.allclose(pred[:7], classify(k, first, mix, penalty, y), atol=1e-10, rtol=0)
        for order in ORDERS:
            output[f'{arm}|{order}'] = pred.argmax(-1).cpu().numpy()
    root = Path(str(seed))
    root.mkdir(exist_ok=True)
    file = root / (held + '.npz')
    np.savez_compressed(file, **output)
    import hashlib
    record = dict(held=held, train_people=sorted(set(people_fit)), source_sha256=sha(Path('source.sha256')),
                  train_rows_sha256=hashlib.sha256(fit.tobytes()).hexdigest(), order_sha256=ORDER_SHA,
                  outputs_sha256=sha(file), weights=learned.tolist(), seed=seed,
                  seconds=time.monotonic()-start, device=torch.cuda.get_device_name(),
                  guard=dict(prefix=True, reset=True, step_equivalence=True, active=True, finite=True))
    (root / (held+'.json')).write_text(json.dumps(record, indent=2))


if __name__ == '__main__':
    main(sys.argv[1], int(sys.argv[2]), sys.argv[3:] == ['smoke'])
