"""Optimize unchanged raw S1C with paired geometric supervision."""
from copy import deepcopy

import numpy as np
import torch
import torch.nn.functional as F

from matched_pairs import relation
from relational import BETA, WEIGHT


def logits(model, z):
    return model.task(torch.cat([z, z.new_zeros((len(z), 287))], 1))


def train(model, x, y, mu, sd, weights, batches, doses, mode, seed):
    model.moment_mean.copy_(mu)
    model.moment_sd.copy_(sd)
    opt = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.001)
    n = sum(map(len, batches))
    scheduler = torch.optim.lr_scheduler.OneCycleLR(opt,.001,total_steps=n)
    torch.manual_seed(seed)
    model.train()
    first = last = None
    maximum = 0.
    activity, proof = [], {}
    for r, dose in zip((r for e in batches for r in e), doses, strict=True):
        opt.zero_grad(set_to_none=True)
        z = model.representation(x[r])
        loss = F.cross_entropy(logits(model, z), y[r], weight=weights)
        if dose is not None:
            indices, literal, sham = dose
            target = literal if mode == "literal" else sham
            if target.requires_grad:
                raise ValueError("teacher target was not detached")
            extra = F.smooth_l1_loss(relation(z, indices), target, beta=BETA)
            if not activity:
                original = model.body.spatial_conv[0].parametrizations.weight.original
                measured, features = torch.autograd.grad(extra, (original, z), retain_graph=True)
                proof.update(auxiliary_gradient=float(measured.abs().max()), feature_gradient=float(features.abs().max()))
                if min(proof.values()) <= 1e-8 or not np.isfinite(list(proof.values())).all():
                    raise ValueError("auxiliary did not genuinely train source waveform features")
            loss = loss + WEIGHT * extra
            activity.append(extra.detach())
        if not torch.isfinite(loss):
            raise ValueError("nonfinite training objective")
        loss.backward()
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        if not grads or any(not g.isfinite().all() for g in grads):
            raise ValueError("invalid native waveform gradients")
        maximum = max(maximum, max(float(g.abs().max()) for g in grads))
        opt.step()
        scheduler.step()
        last = float(loss.detach())
        first = last if first is None else first
    model.eval()
    return dict(steps=n, max_gradient=maximum, first_loss=first, last_loss=last, actual_pairs=len(activity),
                auxiliary_mean=float(torch.stack(activity).mean()), **proof)


@torch.no_grad()
def diagnostic(model, x, flat, doses, selected):
    probe = deepcopy(model).train()
    errors = []
    for step in selected:
        indices, native, sham = doses[step]
        measured = relation(probe.representation(x[flat[step]]), indices)
        errors.append(F.smooth_l1_loss(measured, native, beta=BETA))
    value = float(torch.stack(errors).mean())
    if not np.isfinite(value) or not value > 0:
        raise ValueError("source-only true relation diagnostic was inactive")
    return value
