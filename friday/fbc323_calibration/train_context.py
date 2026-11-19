"""Keep native CE and EMA writes while supervising matched counterfactual predictions."""
from copy import deepcopy

import numpy as np
import torch
import torch.nn.functional as F

from calibration import GROUP, QUERY, STRENGTH, auxiliary, capture, normalize, readout
from source_targets import normalizer


def train(model, x, y, mu, sd, weights, schedule, selections, seed, kind):
    layer = normalizer(model)
    model.moment_mean.copy_(mu)
    model.moment_sd.copy_(sd)
    opt = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.001)
    n = len(schedule)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, .001, total_steps=n)
    torch.manual_seed(seed)
    model.train()
    maximum = first = last = None
    proof = {}
    divergences, movements = [], []
    for rows, eligible in zip(schedule, selections, strict=True):
        opt.zero_grad(set_to_none=True)
        if eligible:
            pred, raw = capture(model, layer, x[rows])
            means = layer.running_mean.detach().clone()
            variance = layer.running_var.detach().clone()
            extra, movement = auxiliary(model, layer, raw, pred, kind)
            if not torch.equal(means, layer.running_mean) or not torch.equal(variance, layer.running_var):
                raise ValueError("counterfactual normalization wrote native running state")
            if not divergences:
                with torch.no_grad():
                    recovered = readout(model, normalize(layer, raw, "native"))
                    delta = float((recovered - pred).abs().max())
                spatial = model.body.spatial_conv[0].parametrizations.weight.original
                affine_gradient, spatial_gradient, raw_gradient = torch.autograd.grad(
                    extra, (layer.weight, spatial, raw), retain_graph=True)
                group_gradient = raw_gradient.reshape(2, GROUP, 288, 1, 480)
                proof = dict(native_delta=delta, aux_affine=float(affine_gradient.abs().max()),
                             aux_spatial=float(spatial_gradient.abs().max()),
                             donor_gradient_nonzero=int(group_gradient[:, QUERY:].count_nonzero()),
                             query_gradient_max=float(group_gradient[:, :QUERY].abs().max()))
                if (not 0 <= delta <= 2e-5 or proof["donor_gradient_nonzero"] or
                        min(proof[k] for k in ("aux_affine", "aux_spatial", "query_gradient_max")) <= 1e-8):
                    raise ValueError(f"native counterfactual readout or auxiliary gradient differs: {proof}")
            divergences.append(extra.detach())
            movements.append(movement)
        else:
            pred, extra = model(x[rows]), 0.
        loss = F.cross_entropy(pred, y[rows], weight=weights) + STRENGTH * extra
        if not torch.isfinite(loss):
            raise ValueError("nonfinite matched source objective")
        loss.backward()
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        if not grads or any(not g.isfinite().all() for g in grads):
            raise ValueError("invalid robust source gradients")
        gradient = max(float(g.abs().max()) for g in grads)
        maximum = max(maximum or 0., gradient)
        opt.step()
        sched.step()
        last = float(loss.detach())
        first = last if first is None else first
    model.eval()
    proof.update(steps=n, first_loss=first, last_loss=last, max_gradient=maximum,
                 aux_steps=len(divergences), aux_queries=len(divergences) * 2 * QUERY,
                 aux_mean=float(torch.stack(divergences).mean()), aux_movement=float(torch.stack(movements).max()))
    if not np.isfinite(list(proof.values())).all() or proof["aux_mean"] <= 0 or proof["aux_movement"] <= 1e-4:
        raise ValueError("actual calibration contexts did not change source predictions")
    return proof


@torch.no_grad()
def diagnose(model, x, schedule, picked):
    clone = deepcopy(model).train()
    layer = normalizer(clone)
    native, within, cross = [], [], []
    for idx in picked:
        logits, raw = capture(clone, layer, x[schedule[idx]])
        native.append(logits.reshape(2, GROUP, 3)[:, :QUERY].reshape(-1, 3))
        within.append(readout(clone, normalize(layer, raw, "within")))
        cross.append(readout(clone, normalize(layer, raw, "cross")))
    p, w, c = [torch.cat(m).double().softmax(1) for m in (native, within, cross)]

    def js(a, b):
        half = .5 * (a + b)
        return float((.5 * (a * (a.log() - half.log()) + b * (b.log() - half.log()))).sum(1).mean())
    values = dict(cross_js=js(p, c), within_js=js(p, w), between_js=js(w, c))
    if not np.isfinite(list(values.values())).all() or min(values.values()) <= 0:
        raise ValueError("detached clean source diagnostic had no measurable context effect")
    return values
