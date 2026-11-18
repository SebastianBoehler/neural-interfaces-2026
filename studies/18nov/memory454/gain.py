"""Pair clean source episodes with coherent or window-independent spectral gains."""
import sys

import torch
from torch.nn import functional as F

from fit import BANDS, fit


def consistency(model, kind, raw, scale, labels, weight, clean):
    assert kind in ('coherent', 'scattered')
    n = 1 if kind == 'coherent' else len(raw)
    offset, tilt = torch.randn(2, n, 1, 41, device=raw.device)
    positions = torch.linspace(-1, 1, len(BANDS), device=raw.device).view(1,-1,1)
    distortion = (.4 * offset + .15 * positions * tilt).flatten(1)
    if len(raw) > 1:
        assert (n == 1) == (kind == 'coherent')
    encoded = model.encoder((raw + distortion / scale).clamp(-8, 8))
    logits, _ = model.sequence(encoded, model.writes(encoded))
    divergence = F.kl_div(logits.log_softmax(-1), clean.detach().softmax(-1), reduction='none').sum(-1)
    return ((F.cross_entropy(logits, labels, reduction='none') + .5 * divergence) * weight).mean()


if __name__ == '__main__':
    fit(sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4:] == ['smoke'], consistency)
