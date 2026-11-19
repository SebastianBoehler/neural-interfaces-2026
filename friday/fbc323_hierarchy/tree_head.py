"""A two-affine-logit categorical likelihood on the original waveform body."""
import math
from types import MethodType

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class Hierarchy(nn.Module):
    def __init__(self, task, width):
        super().__init__()
        if task.weight.shape != (3, width + 287) or task.bias.shape != (3,):
            raise ValueError("original zero-feature flat head differs")
        w, b = task.weight.detach()[:, :width], task.bias.detach()
        motor_prior = b[0] - .5 * (b[1] + b[2]) - math.log(2.)
        self.weight = nn.Parameter(torch.stack((w[0] - .5 * (w[1] + w[2]), w[1] - w[2])))
        self.bias = nn.Parameter(torch.stack((motor_prior, b[1] - b[2])))

    def forward(self, waveform):
        return F.linear(waveform, self.weight, self.bias)


def log_prob(logits):
    motor, subtraction = logits.unbind(1)
    cognitive = F.logsigmoid(-motor)
    return torch.stack((F.logsigmoid(motor), cognitive + F.logsigmoid(subtraction),
                        cognitive + F.logsigmoid(-subtraction)), dim=1)


def install(model):
    model.task = Hierarchy(model.task, model.dim)
    model.forward = MethodType(lambda actor, raw: log_prob(actor.task(actor.representation(raw))), model)
    return model


def verify_loss(model, x, y, w):
    with torch.no_grad():
        logits = model.task(model.representation(x)).detach()
    logits.requires_grad_(True)
    probability = log_prob(logits)
    torch.testing.assert_close(probability.exp().sum(1), torch.ones_like(y, dtype=probability.dtype),
                               rtol=0, atol=3e-7)
    gate = F.binary_cross_entropy_with_logits(logits[:, 0], (y == 0).to(logits.dtype), reduction="none")
    conditional = F.binary_cross_entropy_with_logits(logits[:, 1], (y == 1).to(logits.dtype), reduction="none")
    explicit = ((gate + torch.where(y == 0, 0., conditional)) * w[y]).sum() / w[y].sum()
    actual = F.nll_loss(probability, y, weight=w)
    torch.testing.assert_close(explicit, actual, rtol=0, atol=2e-6)
    motor = torch.autograd.grad(-probability[y == 0, 0].sum(), logits, retain_graph=True)[0]
    cognitive = torch.autograd.grad(-probability[y != 0, :].gather(1, y[y != 0, None]).sum(), logits)[0]
    if (torch.count_nonzero(motor[:, 1]) or not torch.count_nonzero(motor[:, 0]) or
            not torch.count_nonzero(cognitive[:, 1])):
        raise ValueError("hierarchy class routing does not isolate conditional cognition")
    return dict(nll_error=float((actual - explicit).detach().abs()), motor_conditional_gradient=0.)


def train(model, x, y, w, batches, seed):
    """Use the original model322 AdamW/OneCycle recipe with categorical NLL."""
    opt = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.001)
    steps = sum(map(len, batches))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, .001, total_steps=steps)
    torch.manual_seed(seed)
    model.train()
    maximum, first = 0., None
    head_active = torch.zeros(2, device=x.device)
    body_active = 0.
    for epoch in batches:
        for rows in epoch:
            opt.zero_grad(set_to_none=True)
            value = F.nll_loss(model(x[rows]), y[rows], weight=w)
            if not torch.isfinite(value):
                raise ValueError("hierarchical source loss diverged")
            value.backward()
            grads = [p.grad for p in model.parameters() if p.grad is not None]
            if not grads or not all(torch.isfinite(g).all() for g in grads):
                raise ValueError("invalid hierarchical gradient")
            head_active = torch.maximum(head_active, model.task.weight.grad.detach().abs().amax(1))
            if first is None:
                body_active = max(float(p.grad.abs().max()) for p in model.body.parameters() if p.grad is not None)
            maximum = max(maximum, max(float(g.abs().max()) for g in grads))
            opt.step()
            sched.step()
            last = float(value.detach())
            first = last if first is None else first
    model.eval()
    active = [float(v) for v in head_active]
    if min(*active, body_active) <= 0 or not all(np.isfinite((*active, body_active, first, last, maximum))):
        raise ValueError("waveform body or independent affine row did not train")
    return dict(steps=steps, max_gradient=maximum, first_loss=first, last_loss=last,
                motor_gradient=active[0], conditional_gradient=active[1], body_gradient=body_active)
