"""Source-only joint cognitive supervision; auxiliary BN buffers never update."""
import copy
import hashlib
import time

import numpy as np
import torch
from torch import nn

from encoder import Encoder, fingerprint, moments


def buffers(net):
    return {k: v.detach().clone() for k, v in net.named_buffers()}


def auxiliary(net, head, x, labels, transfer):
    """Compute full auxiliary backward; discard encoder gradients in no-transfer arm."""
    previous = net.body.training
    before = buffers(net.body)
    net.body.eval()
    if any(m.training for m in net.body.modules() if isinstance(m, nn.modules.batchnorm._BatchNorm)):
        raise ValueError('auxiliary BatchNorm is training')
    logits = head(net.representation(x))
    loss = nn.functional.cross_entropy(logits, labels)
    loss.backward()
    gradient = sum(float(p.grad.square().sum()) for p in net.body.parameters() if p.grad is not None)
    if not np.isfinite(gradient) or gradient <= 0:
        raise ValueError('inactive auxiliary encoder gradient')
    if not all(torch.equal(v, dict(net.body.named_buffers())[k]) for k, v in before.items()):
        raise ValueError('auxiliary forward changed encoder buffers')
    if not transfer:
        for p in net.body.parameters():
            p.grad = None
    effective = sum(float(p.grad.square().sum()) for p in net.body.parameters() if p.grad is not None)
    if not transfer and effective != 0:
        raise ValueError('stopped encoder gradient is nonzero')
    net.body.train(previous)
    if net.body.training != previous:
        raise ValueError('auxiliary mode restoration failed')
    if not any(p.grad is not None and float(p.grad.abs().max()) > 0 for p in head.parameters()):
        raise ValueError('auxiliary head has no active gradient')
    return float(loss.detach()), gradient, effective


def evaluate(net, x, labels):
    net.eval()
    with torch.no_grad():
        logits = torch.cat([net(b) for b in x.split(64)])
        loss = float(nn.functional.cross_entropy(logits, labels))
        pred = logits.argmax(1)
        ba = float(torch.stack([(pred[labels == c] == c).float().mean() for c in range(3)]).mean())
    return loss, ba


def train(arm, x, y, vx, vy, sx, sy, ss, svx, svy, epochs):
    started = time.monotonic()
    net = Encoder('fb', 0, 'cpu')
    head = nn.Linear(net.dim, 2)
    initial = fingerprint(net)
    initial_aux = fingerprint(head)
    with torch.no_grad():
        z = moments(x)
        net.moment_mean.copy_(z.mean(0)); net.moment_sd.copy_(z.std(0).clamp_min(1e-3))
    main = torch.optim.AdamW(list(net.body.parameters()) + list(net.task.parameters()), lr=.001, weight_decay=.001)
    aux = torch.optim.AdamW(list(net.body.parameters()) + list(head.parameters()), lr=.001, weight_decay=.001)
    gen = torch.Generator().manual_seed(317)
    extra_gen = torch.Generator().manual_seed(318)
    rng = np.random.default_rng(317)
    people = sorted(set(ss.tolist()))
    class_weight = len(y) / (3 * torch.bincount(y, minlength=3).float())
    logs, best, state, max_grad, zero_steps = [], float('inf'), None, 0., 0
    plan = hashlib.sha256()
    for ep in range(epochs):
        net.train()
        for rows in torch.randperm(len(x), generator=gen).split(64):
            plan.update(rows.numpy().tobytes())
            main.zero_grad(set_to_none=True); aux.zero_grad(set_to_none=True)
            loss = nn.functional.cross_entropy(net(x[rows]), y[rows], weight=class_weight)
            loss.backward(); main.step()
            main.zero_grad(set_to_none=True); aux.zero_grad(set_to_none=True)
            if arm == 'A':
                extra = torch.randint(len(x), (len(rows),), generator=extra_gen)
                extra_loss = nn.functional.cross_entropy(net(x[extra]), y[extra], weight=class_weight)
                extra_loss.backward(); main.step()
            else:
                # Exactly one window from a sampled source trial; cache is restricted to win0.
                sample = []
                for _ in range(len(rows)):
                    p = people[rng.integers(len(people))]
                    sample.append(rng.choice(np.flatnonzero(ss == p)))
                before = fingerprint(net.body)
                # Fixed .25 coefficient enters gradients, not merely reported loss.
                l, raw_grad, effective = auxiliary(net, head, sx[sample], sy[sample], arm == 'C')
                for p in list(net.body.parameters()) + list(head.parameters()):
                    if p.grad is not None:
                        p.grad.mul_(.25)
                aux.step()
                if arm == 'B':
                    zero_steps += 1
                    if fingerprint(net.body) != before:
                        raise ValueError('no-transfer auxiliary update changed encoder')
                max_grad = max(max_grad, effective)
        val_loss, ba = evaluate(net, vx, vy)
        net.body.eval(); head.eval()
        with torch.no_grad():
            outputs = torch.cat([head(net.representation(b)) for b in svx.split(64)])
            predicted = outputs.argmax(1)
            auxiliary_ba = float(torch.stack([(predicted[svy == c] == c).float().mean() for c in (0, 1)]).mean())
        logs.append(dict(epoch=ep + 1, loss=val_loss, ba=ba, auxiliary_ba=auxiliary_ba))
        print(arm, logs[-1], flush=True)
        if not np.isfinite(val_loss):
            raise ValueError('nonfinite source validation')
        if val_loss < best:
            best, state = val_loss, copy.deepcopy(net.state_dict())
        # Stop immediately on material training collapse, never consult outer data.
        if ep >= 4 and val_loss > 5:
            raise ValueError('source pilot collapse')
    net.load_state_dict(state); net.eval()
    selected_loss, selected_ba = evaluate(net, vx, vy)
    if arm == 'C' and max_grad <= 0:
        raise ValueError('joint supervision never reached encoder')
    return dict(arm=arm, seconds=time.monotonic()-started, initial=initial,
                initial_aux=initial_aux, final=fingerprint(net), source_plan=plan.hexdigest(),
                selected_source_loss=selected_loss, selected_source_ba=selected_ba,
                auxiliary_max_effective_gradient=max_grad, stopped_steps=zero_steps, epochs=logs)
