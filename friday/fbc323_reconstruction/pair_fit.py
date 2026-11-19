"""Train the native S1C with a detached, isolated reconstruction companion."""
import numpy as np
import torch
from torch.nn import functional as F

from source_targets import normalizer, raw_logits


def two_views(x, views, row, code):
    first = x[row].clone()
    second = first.clone()
    for i, own in enumerate(views):
        positions = np.flatnonzero(code == i)
        first[positions] = own[row[positions]]
        second[positions] = views[1 - i][row[positions]]
    return first, second


def companion(model, layer, x):
    mean, var = layer.running_mean.clone(), layer.running_var.clone()
    if not layer.training or layer.causal or len(x) % 32:
        raise ValueError("companion must use the native session-pure training normalization")
    try:
        with torch.no_grad(), torch.random.fork_rng(devices=[x.device.index or 0]):
            logits = raw_logits(model, x)
    finally:
        layer.running_mean.copy_(mean)
        layer.running_var.copy_(var)
    if logits.requires_grad:
        raise ValueError("companion target was not detached")
    return logits


def train(model, x, y, mu, sd, weights, batches, codes, views, strength, seed):
    model.moment_mean.copy_(mu)
    model.moment_sd.copy_(sd)
    opt = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.001)
    flat = [r for epoch in batches for r in epoch]
    steps = len(flat)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, .001, total_steps=steps)
    torch.manual_seed(seed)
    model.train()
    bn = normalizer(model)
    maximum, first, counts, measured_loss = 0., None, [0, 0], []
    evidence = {}
    for i, (row, code) in enumerate(zip(flat, codes, strict=True)):
        opt.zero_grad(set_to_none=True)
        primary, opposite = two_views(x, views, row, code)
        selected = code >= 0
        if i == 0:
            native_in = x[row]
            if (not torch.equal(primary[~selected], native_in[~selected]) or
                    not torch.equal(opposite[~selected], native_in[~selected]) or
                    float((primary[selected] - opposite[selected]).abs().max()) <= 1e-4):
                raise ValueError("source input pairing is not active and exact")
            rm, rv = bn.running_mean.clone(), bn.running_var.clone()
            cpu, gpu = torch.random.get_rng_state(), torch.cuda.get_rng_state()
        target = companion(model, bn, opposite)
        if i == 0 and (not torch.equal(rm, bn.running_mean) or not torch.equal(rv, bn.running_var) or
                not torch.equal(cpu, torch.random.get_rng_state()) or not torch.equal(gpu, torch.cuda.get_rng_state())):
            raise ValueError("companion changed primary running statistics or RNG")
        logits = raw_logits(model, primary)
        ce = F.cross_entropy(logits, y[row], weight=weights)
        kld = F.kl_div(F.log_softmax(logits[selected], 1), F.softmax(target[selected], 1), reduction="batchmean")
        if i == 0:
            spatial = model.body.spatial_conv[0].parametrizations.weight.original
            bg, hg = torch.autograd.grad(kld, (spatial, model.task.weight), retain_graph=True)
            if (not np.isfinite([float(bg.norm()), float(hg.norm()), float(kld.detach())]).all() or
                    torch.count_nonzero(hg[:, -287:]) or not torch.count_nonzero(bg) or not torch.count_nonzero(hg) or
                    not kld.detach() > 0):
                raise ValueError("actual detached KL did not reach the raw body and classifier")
            evidence = dict(kl_body_gradient=float(bg.norm()), kl_head_gradient=float(hg.norm()),
                            companion_isolated=True, kl_first=float(kld.detach()))
        loss = ce + strength * kld if strength else ce
        if not torch.isfinite(loss):
            raise ValueError("reconstruction source loss is nonfinite")
        loss.backward()
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        if not grads or not all(torch.isfinite(g).all() for g in grads):
            raise ValueError("actual source gradients are invalid")
        maximum = max(maximum, max(float(g.abs().max()) for g in grads))
        opt.step()
        sched.step()
        last = float(loss.detach())
        first = last if first is None else first
        measured_loss.append(kld.detach())
        for j in (0, 1):
            counts[j] += int((code == j).sum())
    model.eval()
    if counts[0] != counts[1] or sum(counts) * 4 != sum(map(len, flat)) or maximum <= 0:
        raise ValueError("physical primary reconstruction exposure is unpaired")
    return dict(steps=steps, first_loss=first, last_loss=last, max_gradient=maximum, view_counts=counts,
                mean_kl=float(torch.stack(measured_loss).mean()), **evidence)


@torch.no_grad()
def source_js(model, stream, views, picked):
    post = []
    if len(picked) != 64 or model.training:
        raise ValueError("source diagnostic rows differ")
    for view in views:
        values = torch.zeros((len(view), 3), device=view.device, dtype=torch.float64)
        for name in np.unique(stream[picked]):
            rows = np.flatnonzero(stream == name)
            values[rows] = torch.log_softmax(raw_logits(model, view[rows]).double(), 1)
        post.append(values[picked])
    p, q = post
    average = ((p.exp() + q.exp()) / 2).log()
    value = float((p.exp() * (p - average) + q.exp() * (q - average)).sum(1).mean() / 2)
    if not np.isfinite(value) or value < 0:
        raise ValueError("source pair diagnostic returned invalid divergence")
    return value
