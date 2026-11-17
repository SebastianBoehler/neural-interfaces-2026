"""Fit two matched source readouts and verify transfer with P03 labels unopened."""
import json
import resource
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from inputs import ORDERS, digest, prepare, preflight, require
from readout import Prefix, Readout


def stack(streams):
    return tuple(torch.from_numpy(np.stack([streams[order][i] for order in ORDERS])).double()
                 for i in (0, 1))


def fit(streams, labels, prefix):
    probability, items = stack(streams)
    order_labels = np.stack([labels[streams[order][2]] for order in ORDERS])
    targets = torch.from_numpy(order_labels.astype(np.int64)).flatten()
    weights = torch.bincount(targets, minlength=3)
    require((weights > 0).all(), 'source class missing')
    weights = targets.numel() / (3 * weights.double())
    torch.manual_seed(357)
    model = Readout().double()
    initial = torch.cat([p.detach().ravel() for p in model.parameters()])
    optimizer = torch.optim.Adam(model.parameters(), lr=.003)
    logp = probability.log()
    loss_first = None
    item_grade = 0.
    for _ in range(16):
        logits = model(logp, items, prefix).reshape(-1, 3)
        loss = F.cross_entropy(logits, targets, weight=weights)
        require(torch.isfinite(loss), 'nonfinite source fit')
        if loss_first is None:
            loss_first = float(loss.detach())
        optimizer.zero_grad()
        loss.backward()
        item_grade = max(item_grade, float(model.item[0].weight.grad.abs().max()))
        optimizer.step()
    with torch.no_grad():
        last = float(F.cross_entropy(model(logp, items, prefix).reshape(-1, 3), targets, weight=weights))
        changed = float((torch.cat([p.ravel() for p in model.parameters()]) - initial).abs().max())
    require(item_grade > 1e-8 and changed > 1e-7 and np.isfinite(last), 'source fitting silent no-op')
    return model.eval(), dict(active_gradient_max=item_grade, parameter_change_max=changed,
                             parameters=len(initial), initial_sha=digest(initial.numpy()),
                             source_loss_initial=loss_first, source_loss_final=last, optimizer_steps=16)


@torch.no_grad()
def guards(current, causal, streams, other):
    prefix_outputs = {}
    maxima = dict(vector_vs_step=0., prefix=0., same_set=0., source_interleave=0., prefix_sensitivity=0.)
    for order in ORDERS:
        probability, items = (torch.from_numpy(x).double() for x in streams[order][:2])
        logp = probability.log()
        fixed = current.replay(probability, items, prefix=False)
        replay = causal.replay(probability, items, prefix=True)
        require(torch.equal(causal.replay(probability, items, prefix=True, active=False), probability),
                'disabled readout fails exact historical identity')
        for model, prefix, expected in ((current, False, fixed), (causal, True, replay)):
            actual = model(logp[None], items[None], prefix).softmax(-1)[0]
            err = float((actual - expected).abs().max())
            maxima['vector_vs_step'] = max(maxima['vector_vs_step'], err)
            require(err <= 1e-12 and torch.isfinite(expected).all()
                    and torch.allclose(expected.sum(-1), torch.ones(len(items), dtype=torch.float64), atol=1e-12, rtol=0),
                    'train/step probability mismatch')
        cut = 17
        prefix_out = causal.replay(probability[:cut], items[:cut], prefix=True)
        maxima['prefix'] = max(maxima['prefix'], float((prefix_out - replay[:cut]).abs().max()))
        shuffled = items.clone()
        shuffled[:cut] = shuffled[:cut].flip(0)
        same = causal.replay(probability[:cut+1], shuffled[:cut+1], prefix=True)[cut]
        maxima['same_set'] = max(maxima['same_set'], float((same - replay[cut]).abs().max()))
        changed = items.clone()
        changed[:cut].zero_()
        sensitivity = (causal.replay(probability[:cut+1], changed[:cut+1], prefix=True)[cut] - replay[cut]).abs().max()
        maxima['prefix_sensitivity'] = max(maxima['prefix_sensitivity'], float(sensitivity))
        current_changed = items.clone()
        current_changed[cut].zero_()
        strictly_prior = causal.replay(probability[:cut+1], current_changed[:cut+1], prefix=True)[cut]
        require(torch.equal(strictly_prior, replay[cut]), 'current item entered its own support set')
        current_control = current.replay(probability, shuffled, prefix=False)[cut]
        require(torch.equal(current_control, fixed[cut]), 'current-only control used prior items')
        prefix_outputs[order] = replay
    p, x = (torch.from_numpy(v).double() for v in streams['within0'][:2])
    q, y = (torch.from_numpy(v).double() for v in other['within0'][:2])
    first, second = Prefix(causal), Prefix(causal)
    interleaved = []
    for a, b, c, d in zip(p, x, q, y):
        interleaved.append(first.step(a, b))
        second.step(c, d)
    maxima['source_interleave'] = float((torch.stack(interleaved) - prefix_outputs['within0']).abs().max())
    first.reset()
    require(torch.equal(torch.stack([first.step(a, b) for a, b in zip(p[:17], x[:17])]), prefix_outputs['within0'][:17]),
            'session reset replay differs')
    require(maxima['same_set'] <= 1e-12 and maxima['prefix'] == 0 and maxima['source_interleave'] == 0
            and maxima['prefix_sensitivity'] >= 1e-6, 'unordered/prefix/activity/interleave guard')
    return maxima


def main():
    start = time.monotonic()
    torch.set_num_threads(2)
    data, labels, trace, base_hash = prepare()
    current, current_trace = fit(data['P02'], labels, prefix=False)
    causal, causal_trace = fit(data['P02'], labels, prefix=True)
    require(current_trace['initial_sha'] == causal_trace['initial_sha']
            and current_trace['parameters'] == causal_trace['parameters'], 'matched-capacity control differs')
    guards_out = guards(current, causal, data['P03'], data['P02'])
    preflight()
    proof = dict(status='SOURCE_ONLY_READINESS_PASS', source_trace=trace, body_sha=base_hash,
                 fit_people=['P02'], guard_people=['P03'], p03_labels_opened=False,
                 p01_p09_neural_forward=False, quality_scored=False, orders=list(ORDERS),
                 matched_current=current_trace, learned_prefix=causal_trace, guards=guards_out,
                 wall_s=time.monotonic() - start, rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    with Path('proof.json').open('x') as f:
        json.dump(proof, f, indent=2)
    print(json.dumps(proof), flush=True)


if __name__ == '__main__':
    main()
