"""Reconstruct the owning frozen ocular embedding without changing its weights."""

from collections import namedtuple
import runpy

import numpy as np
import torch

from eye_scope import OCUNET


EyeSeed = namedtuple("EyeSeed", ["network", "trace"])


def load_networks(records):
    constructor = runpy.run_path(str(OCUNET))["Net"]
    out = []
    for item in records:
        trace = torch.jit.load(item["path"], map_location="cpu").eval()
        net = constructor().cpu().eval()
        net.load_state_dict(trace.state_dict(), strict=True)
        if net.fc.in_features != 128 or net.fc.out_features != 3:
            raise ValueError("the original owning Eye embedding or classes differ")
        out.append(EyeSeed(net, trace))
    return tuple(out)


@torch.inference_mode()
def embeddings(nets, x, check=False):
    tensors = torch.from_numpy(np.ascontiguousarray(x))
    outputs, peak = [], 0.
    for net, trace in nets:
        h = net.embed(tensors)
        if h.shape != (len(x), 128) or not torch.isfinite(h).all():
            raise ValueError("invalid actual Eye embedding")
        if check:
            reconstructed, original = net.fc(h), trace(tensors)
            error = float((reconstructed - original).abs().max())
            if not 0 <= error <= 1e-5 or not torch.equal(reconstructed.argmax(1), original.argmax(1)):
                raise ValueError(f"original owning Eye trace does not match its reconstruction: {error}")
            peak = max(peak, error)
        outputs.append(h.to(torch.float64).numpy())
    return np.concatenate(outputs, axis=1), peak
