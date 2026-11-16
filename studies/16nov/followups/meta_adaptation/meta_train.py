"""Fit source-only update multipliers through genuine differentiable head steps."""
import json
import os
import sys
import time
from pathlib import Path
import numpy as np
import torch

ROOT = Path(os.environ['NI26_WS']).resolve(strict=True)
os.environ['NI26_ROOT'] = str(ROOT)
sys.path[:0] = [str(Path(__file__).resolve().parent), str(ROOT / 'experiments')]
import protocol
from adapt import FEATURES, BLOCKS, update

ORDERS = ('within0','full0','runs0')


def load(path, d, outer, source, smoke):
    with np.load(path, allow_pickle=False) as z:
        meta = json.loads(str(z['metadata'].item()))
        expected = [f'P{i:02d}' for i in range(1,11) if f'P{i:02d}' not in (outer,source)]
        if meta['outer'] != outer or meta['source'] != source or meta['fit_people'] != expected or meta['smoke'] != smoke:
            raise ValueError('crossfit provenance or smoke mismatch')
        fit_rows, source_rows = z['fit_rows'].copy(), z['source_rows'].copy()
        if set(d['subject'][fit_rows]) != set(expected) or set(d['subject'][source_rows]) != {source}:
            raise ValueError('outer/source exclusion failed')
        values = {k: z[k].copy() for k in ('theta','mean','scale')}
        if values['theta'].shape != (3, FEATURES + 1) or np.any(values['scale'] <= 0):
            raise ValueError('invalid fitted source anchor')
        values['rows'] = source_rows
        values['orders'] = {}
        for name in ORDERS[:1] if smoke else ORDERS:
            ordered = z[f'order|{name}'].copy()
            feat = z[f'features|{name}'].copy()
            if feat.shape != (len(source_rows), FEATURES) or not np.array_equal(np.sort(ordered), source_rows):
                raise ValueError('source stream rows differ')
            values['orders'][name] = (ordered, feat)
    return values


def episodes(data, d, epoch, seed, smoke):
    result = []
    for person in sorted(data):
        sample = data[person]; seen = 0
        for name, (order, features) in sample['orders'].items():
            for ordinal, session in enumerate(dict.fromkeys(d['stream'][order])):
                rows = order[d['stream'][order] == session]
                cuts = list(range(40, len(rows), 20))
                if not cuts: continue
                t = cuts[(epoch + ordinal + seed) % len(cuts)]
                before, after = rows[max(0,t-64):t], rows[t:min(len(rows), t+20)]
                if not len(after) or set(before) & set(after): raise ValueError('invalid source support/query')
                result.append((person,name,session, before,after,features)); seen += 1
        if not seen: raise ValueError(f'{person}: no eligible source episodes')
    rng = np.random.default_rng(seed * 1000 + epoch)
    result = [result[i] for i in rng.permutation(len(result))]
    return result[:1] if smoke else result


def fit(outer, kind, seed, smoke=False):
    if kind not in ('scalar','blocks') or seed not in (0,1): raise ValueError('invalid meta arm')
    target = ROOT / f'runs/s307/meta/{outer}_{kind}_s{seed}{"_smoke" if smoke else ""}.npz'
    if target.exists(): raise FileExistsError(target)
    d = protocol.load()
    people = [f'P{i:02d}' for i in range(1,11) if f'P{i:02d}' != outer]
    if smoke: people = people[:1]
    data = {p: load(ROOT / f'runs/s307/crossfit/{outer}_{p}.npz', d, outer, p, smoke) for p in people}
    device = torch.device('cuda:0')
    init = np.arctanh(.5)
    parameter = torch.full(() if kind == 'scalar' else (BLOCKS,), init, dtype=torch.float64, device=device, requires_grad=True)
    optimizer = torch.optim.Adam([parameter], lr=.01)
    initial = parameter.detach().clone(); total, max_grad, start = 0, 0., time.monotonic()
    for epoch in range(1 if smoke else 12):
        for person,name,session,support,query,features in episodes(data,d,epoch,seed,smoke):
            source = data[person]; loc = np.searchsorted(source['rows'], np.r_[support,query])
            x = (features[loc] - source['mean']) / source['scale']
            if not np.isfinite(x).all(): raise ValueError('nonfinite standardized support/query')
            x = torch.as_tensor(x, dtype=torch.float64, device=device)
            anchor = torch.as_tensor(source['theta'][:,:-1], dtype=torch.float64, device=device)
            bias = torch.as_tensor(source['theta'][:,-1], dtype=torch.float64, device=device)
            target_y = torch.as_tensor(d['y'][query], dtype=torch.int64, device=device)
            optimizer.zero_grad()
            head, stats = update(anchor, anchor.clone(), bias, x[:len(support)], parameter.tanh(), create_graph=True)
            value = torch.nn.functional.cross_entropy(x[len(support):] @ head.T + bias, target_y)
            value.backward()
            if parameter.grad is None or not torch.isfinite(parameter.grad).all().item(): raise ValueError('missing meta gradient')
            g = float(parameter.grad.abs().max())
            if g <= 0: raise ValueError('disconnected or zero meta gradient')
            optimizer.step(); max_grad = max(max_grad,g); total += 1
            if stats['gradient_max'] <= 0 or stats['delta_ratio_max'] > .03000001: raise ValueError('inner update guard failed')
    if not total or torch.equal(initial, parameter.detach()): raise ValueError('meta parameters did not change')
    target.parent.mkdir(parents=True, exist_ok=True)
    metadata = {'outer': outer,'kind': kind,'seed': seed,'episodes':total,'meta_gradient_max':max_grad,
                'seconds':time.monotonic()-start,'source_people':people,'smoke':smoke,'held_accuracy_computed':False}
    with target.open('xb') as f:
        np.savez(f, parameter=parameter.detach().cpu().numpy(), metadata=json.dumps(metadata,sort_keys=True))
    print(f'{outer}/{kind}/{seed}: {total} source-only episodes; nonzero meta gradient {max_grad:.6g}; no held score',flush=True)


if __name__ == '__main__': fit(sys.argv[1],sys.argv[2],int(sys.argv[3]),len(sys.argv) == 5 and sys.argv[4] == 'smoke')
