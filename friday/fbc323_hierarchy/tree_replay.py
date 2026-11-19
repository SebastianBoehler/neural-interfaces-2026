"""Score the frozen two-logit S1C as one three-class raw decoder."""
from types import MethodType

import numpy as np
import torch

from bounded_bn import normalize
from encoder312 import fingerprint
from replay_f13 import ORDERS
from source_targets import TOL, normalizer
from tree_head import install


def frozen_tree(path, meta):
    import model322

    model322.SEED = meta["seed"]
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if any(payload[k] != meta[k] for k in ("kind", "seed", "held", "scale")):
        raise ValueError("loaded tree does not identify the authenticated fit")
    model = install(model322.Factor("session", False, "cpu")).eval()
    model.load_state_dict(payload["state"], strict=True)
    layer = normalizer(model)
    if (model.task.weight.shape != (2, model.dim) or fingerprint(model) != meta["state"] or
            sum(p.numel() for p in model.parameters() if p.requires_grad) != meta["parameters"]):
        raise ValueError("fitted raw-only hierarchy or frozen state differs")
    return model, layer


def probability(model, raw):
    return model(torch.from_numpy(np.ascontiguousarray(raw))).double().exp().cpu().numpy()


@torch.no_grad()
def replay_tree(model, layer, x, stream, test, perms, scale):
    before, native = fingerprint(model), layer.forward
    result = np.full((2, len(ORDERS), len(test)), -1, dtype=np.int8)
    counts = []
    bounded = MethodType(lambda this, value: normalize(this, value, counts), layer)
    first = resets = clips = 0
    max_diff = 0.
    for j, order in enumerate(ORDERS):
        position = perms[order]
        for name in dict.fromkeys(stream[test]):
            rows = position[stream[position] == name]
            other = position[stream[position] != name][:10]
            sample, crossed = x[rows] / scale, x[other] / scale
            for m, call in enumerate((native, bounded)):
                layer.forward = call
                counts.clear()
                p = probability(model, sample)
                if p.shape != (len(rows), 3) or not np.isfinite(p).all() or not np.allclose(p.sum(1), 1, atol=3e-7, rtol=0):
                    raise ValueError("hierarchical heads did not produce normalized probabilities")
                if m == 0:
                    start = p[0].copy()
                else:
                    if len(counts) != 1 or not np.array_equal(start, p[0]):
                        raise ValueError("hierarchical bounded normalization changed the first raw window")
                    clips += int(counts[0])
                    first += 1
                prefix = probability(model, sample[:10])
                max_diff = max(max_diff, float(np.abs(p[:10] - prefix).max()))
                if not np.allclose(p[:10], prefix, atol=TOL, rtol=TOL) or not np.array_equal(
                        p[:10].argmax(1), prefix.argmax(1)):
                    raise ValueError("later held frames changed the raw-only hierarchy prefix")
                probability(model, crossed)
                if not np.array_equal(probability(model, sample[:10]), prefix):
                    raise ValueError("hierarchical head retained previous session context")
                resets += 1
                result[m, j, np.searchsorted(test, rows)] = p.argmax(1)
            layer.forward = native
    if fingerprint(model) != before or (result < 0).any():
        raise ValueError("held hierarchy changed frozen weights or missed rows")
    return result, dict(first_windows_exact=first, resets=resets, clips=clips, prefix_max_abs=max_diff)
