"""Counterfactual SessionBN outputs using real detached source moments."""
import hashlib
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


HERE = Path(__file__).resolve().parent
KINDS = ("hard", "jitter", "within", "cross")
MODES = ("ordinary", "bounded")
ARMS = tuple(f"{kind}_{mode}" for kind in KINDS for mode in MODES)
GROUP, QUERY, TEMPERATURE, STRENGTH = 32, 8, 2., .20


def plan(batches, streams):
    people = np.asarray([s.split("|", 1)[0] for s in streams])
    rows = [r for epoch in batches for r in epoch]
    selected = []
    for r in rows:
        if len(r) not in (GROUP, 2 * GROUP):
            raise ValueError("original training batch does not use 32-row chunks")
        chunks = r.reshape(-1, GROUP)
        if any(len(set(streams[c])) != 1 for c in chunks):
            raise ValueError("original training chunk crosses source sessions")
        selected.append(bool(len(chunks) == 2 and len(set(people[r])) == 2 and len(set(r)) == len(r)))
    eligible = np.flatnonzero(selected)
    if len(eligible) < 6:
        raise ValueError("not enough disjoint paired source contexts")
    digest = hashlib.sha256(np.asarray(selected, dtype=np.uint8).tobytes()).hexdigest()
    return rows, selected, eligible[:6], digest


def normalize(layer, raw, context):
    if raw.shape != (2 * GROUP, 288, 1, 480):
        raise ValueError("counterfactual source pass requires two true FBCNet chunks")
    if context not in ("within", "cross", "native"):
        raise ValueError("unregistered context")
    group = raw.reshape(2, GROUP, 288, 1, 480)
    query, donor = (group, group) if context == "native" else (group[:, :QUERY], group[:, QUERY:].detach())
    with torch.no_grad():
        mean = donor.mean((3, 4)).mean(1)
        var = (donor.square().mean((3, 4)).mean(1) - mean.square()).clamp_min(0)
        if context == "cross":
            mean, var = mean.flip((0,)), var.flip((0,))
    centered = (query - mean[:, None, :, None, None]) / (var[:, None, :, None, None] + layer.eps).sqrt()
    return layer._affine(centered.flatten(0, 1), (1, -1, 1, 1))


def readout(model, normalized):
    body = model.body
    if len(body.spatial_conv) != 3 or type(body.spatial_conv[2]) is not torch.nn.SiLU:
        raise ValueError("native FBCNet activation/readout geometry differs")
    z = body.padding_layer(body.spatial_conv[2](normalized))
    z = z.reshape(len(z), 288, 4, 120)
    z = body.final_layer(body.flatten_layer(body.temporal_layer(z)))
    if z.shape != (len(normalized), 1152) or model.skip or model.task.in_features != 1439:
        raise ValueError("raw-only native class head width differs")
    return model.task(torch.cat([z, z.new_zeros((len(z), 287))], 1))


def auxiliary(model, layer, raw, native, context):
    expected = native.view(2, GROUP, 3)[:, :QUERY].reshape(-1, 3).detach()
    score = readout(model, normalize(layer, raw, context))
    teacher = (expected / TEMPERATURE).softmax(1)
    loss = F.kl_div((score / TEMPERATURE).log_softmax(1), teacher, reduction="batchmean") * TEMPERATURE**2
    return loss, (score - expected).detach().abs().max()


def capture(model, layer, raw):
    found = []
    hook = layer.register_forward_pre_hook(lambda _, x: found.append(x[0]))
    try:
        logits = model(raw)
    finally:
        hook.remove()
    if len(found) != 1:
        raise ValueError("native actor did not traverse one SessionBN")
    return logits, found[0]
