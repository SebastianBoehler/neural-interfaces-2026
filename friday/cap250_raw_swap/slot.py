"""Replace one non-refitted member at the original NetBank call site."""
import copy
from pathlib import Path

import numpy as np
import torch


MEMBER = 48
ARCHIVE_NAME = "sbn_fbcnet_raw_k20_s1.pt"


class Slot:
    def __init__(self, original, input_stream, actor):
        self.original = original if actor is None else None
        self.input_stream = input_stream
        self.actor = copy.deepcopy(actor) if actor is not None else None
        self.dtype = original.dtype
        self.reset_state()

    def reset_state(self):
        self.window = None
        self.calls = 0
        if self.actor is None:
            self.original.reset_state()
        else:
            self.actor.reset()

    def __call__(self, x):
        if self.window is not None or x.shape != (41, 480) or x.dtype != np.float32:
            raise ValueError("expected one canonical raw window at NetBank")
        self.window = x
        return self.input_stream(x)

    def proba(self, scaled, cache):
        if self.window is None:
            raise ValueError("the native slot did not receive the current window")
        if self.actor is None:
            posterior = self.original.proba(scaled, cache)
        else:
            window = torch.from_numpy(self.window[None]).to(device=scaled.device)
            # Keep CUDA graph profiling from changing a fresh actor copy’s numerics.
            with torch.jit.optimized_execution(False):
                posterior = self.actor(window)
        self.window = None
        self.calls += 1
        if posterior.shape != (1, 3) or posterior.dtype != torch.float64:
            raise ValueError("the replacement must emit the native three-class posterior")
        return posterior


def install(model, actor):
    with np.load(Path("/package/weights.npz"), allow_pickle=False) as z:
        if str(z[f"net{MEMBER}_sjit_file"]) != ARCHIVE_NAME:
            raise ValueError("unexpected cap250 member artifact")
    if (not np.isclose(model.w[MEMBER + 1], 1 / 90, rtol=0, atol=1e-16)
            or model.ft.member["fbraw"] != 47 or MEMBER in model.ft.member.values()
            or list(model.ft.slots["fbraw"]) != list(range(47, 53))):
        raise ValueError("native weight or refitted member assignment changed")
    found = []
    for g, (stream, members) in enumerate(model.nets.groups):
        for j, (i, net) in enumerate(members):
            if i == MEMBER:
                found.append((g, stream, members, j, net))
    if len(found) != 1:
        raise ValueError("expected exactly one live slot")
    g, stream, members, j, net = found[0]
    if type(net).__name__ != "SharedFIRSJit" or net.fblight or stream.align:
        raise ValueError("unexpected incumbent waveform class or input stream")
    slot = Slot(net, stream, actor)
    members[j] = (MEMBER, slot)
    model.nets.groups[g] = (slot, members)
    return slot
