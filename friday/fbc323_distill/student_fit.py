"""Run the original S1C source optimizer with the fixed F16 soft loss."""
import copy

import numpy as np
import torch
from torch.nn import functional as F

from source_targets import raw_logits


TEMPERATURE = 2.
ALPHA = .25


def kl(logits, target):
    return TEMPERATURE ** 2 * F.kl_div(
        F.log_softmax(logits / TEMPERATURE, dim=1), target, reduction="batchmean")


def soft_probe(model, x, mu, sd, sample, target):
    """Prove KL alone reaches the waveform and the classifier, without updating the fit."""
    with torch.random.fork_rng(devices=[x.device.index or 0]):
        probe = copy.deepcopy(model).train()
        probe.moment_mean.copy_(mu)
        probe.moment_sd.copy_(sd)
        out = raw_logits(probe, x[sample])
        loss = kl(out, target)
        loss.backward()
        body = torch.stack([p.grad.norm() for p in probe.body.parameters() if p.grad is not None]).norm()
        head = probe.task.weight.grad
        if torch.count_nonzero(head[:, -287:]).item() != 0:
            raise ValueError("the zero-q feature columns received KL gradient")
        values = dict(kl_body=float(body), kl_head=float(head[:, :-287].norm()), kl=float(loss.detach()))
        if not all(np.isfinite(v) and v > 0 for v in values.values()):
            raise ValueError("trial or deranged target is inactive on the raw body or classifier")
        return values


def train_soft(model, x, y, mu, sd, weights, batches, target, seed):
    model.moment_mean.copy_(mu)
    model.moment_sd.copy_(sd)
    opt = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.001)
    steps = sum(map(len, batches))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, .001, total_steps=steps)
    torch.manual_seed(seed)
    model.train()
    maximum, first = 0., None
    for epoch, doses in zip(batches, target, strict=True):
        offset = 0
        for rows in epoch:
            opt.zero_grad(set_to_none=True)
            logits = raw_logits(model, x[rows])
            end = offset + len(rows)
            soft = kl(logits, doses[offset:end])
            loss = (1 - ALPHA) * F.cross_entropy(logits, y[rows], weight=weights) + ALPHA * soft
            if not torch.isfinite(loss):
                raise ValueError("student loss diverged")
            loss.backward()
            grads = [p.grad for p in model.parameters() if p.grad is not None]
            if not grads or not all(torch.isfinite(g).all() for g in grads):
                raise ValueError("invalid student gradient")
            maximum = max(maximum, max(float(g.abs().max()) for g in grads))
            opt.step()
            sched.step()
            last = float(loss.detach())
            first = last if first is None else first
            offset = end
        if offset != len(doses):
            raise ValueError("distillation targets do not match original source batches")
    model.eval()
    if maximum <= 0:
        raise ValueError("the student did not train")
    return dict(steps=steps, max_gradient=maximum, first_loss=first, last_loss=last)
