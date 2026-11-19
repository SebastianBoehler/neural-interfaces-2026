"""Split the archived spatial trace and preserve the normal cap250 member interface."""
import copy

import numpy as np
import torch


RAW_MEMBERS = tuple(range(47, 53))
TOLERANCE = 2e-5


class BoundedRaw:
    def __init__(self, net, arm):
        self.trace = net.m
        spatial = self.trace.spatial_conv
        self.conv, self.bn, self.activation = (getattr(spatial, str(i)) for i in range(3))
        if (self.trace.original_name, self.bn.original_name, self.activation.original_name) != (
                "FBCNet", "StreamBN", "SiLU") or not torch.equal(self.bn.n, torch.zeros_like(self.bn.n)):
            raise ValueError("expected fresh archived raw FBCNet and StreamBN")
        self.original = copy.deepcopy(spatial) if arm == "zero" else None
        self.arm = arm
        self.n = 0
        self.clipped = []
        self.max_errors = []
        self.checks = []
        self.rm = self.bn.rm.double()
        self.rv = self.bn.rv.double()
        self.mass, self.sum1, self.sum2 = (torch.zeros_like(self.rm) for _ in range(3))
        self.seen_clip = torch.zeros_like(self.rm, dtype=torch.bool)

    def posterior(self, h):
        trace = self.trace
        h = trace.temporal_layer(h.view(1, h.shape[1], 4, 120))
        return torch.softmax(trace.final_layer(trace.flatten_layer(h)).double(), 1)

    def weighted(self, x, stock):
        x64 = x.double()
        r1, r2 = x64.mean((0, 2, 3)), x64.square().mean((0, 2, 3))
        mass = 21 + self.mass
        mean = (20 * self.rm + self.sum1 + r1) / mass
        power = 20 * (self.rv + self.rm.square()) + self.sum2 + r2
        variance = (power / mass - mean.square()).clamp_min(0)
        shape = (1, -1, 1, 1)
        normalized = (x - mean.to(x.dtype).view(shape)) / (variance.to(x.dtype).view(shape) + 1e-5).sqrt()
        affine = normalized * self.bn.w.view(shape) + self.bn.b.view(shape)
        output = torch.where(self.seen_clip.view(shape), affine, stock)
        if not self.n and not torch.equal(output, stock):
            raise ValueError("weighted mean changed the first raw member activation")
        energy = (r2 - 2 * r1 * self.rm + self.rm.square()).clamp_min(1e-5)
        weight = (4 * self.rv.clamp_min(1e-5) / energy).clamp(max=1)
        clipped = weight < 1
        self.clipped.append(torch.count_nonzero(clipped))
        self.mass += weight
        self.sum1 += weight * r1
        self.sum2 += weight * r2
        self.seen_clip |= clipped
        return output

    def __call__(self, x, cache):
        if x.shape != (1, 41, 480):
            raise ValueError("expected the official one-window worker delivery")
        if "sbn_bank" not in cache:
            cache["sbn_bank"] = self.trace.spectral_filtering(x)
        y = cache["sbn_bank"]
        values = self.conv(y)
        stock = self.bn(values)
        normalized = self.weighted(values, stock) if self.arm == "bounded" else stock
        output = self.posterior(self.activation(normalized))
        if self.original is not None:
            reference = self.posterior(self.original(y))
            diff = (output - reference).abs()
            self.max_errors.append(diff.max())
            self.checks.append((diff <= TOLERANCE + TOLERANCE * reference.abs()).all()
                               & (reference.argmax() == output.argmax()))
        self.n += 1
        return output

    def receipt(self):
        bounded = self.arm == "bounded"
        if not np.isclose(float(self.bn.n), self.n, atol=0, rtol=0):
            raise ValueError("archived StreamBN advanced by the wrong number of windows")
        if self.original is not None:
            original = getattr(self.original, "1")
            if not torch.stack(self.checks).all().item() or not torch.equal(original.n, self.bn.n):
                raise ValueError("the split trace changed a raw member posterior or argmax")
        return dict(windows=self.n, clipped=int(torch.stack(self.clipped[:-1]).sum())
                    if bounded and self.n > 1 else 0,
                    max_archive_posterior_difference=float(torch.stack(self.max_errors).max())
                    if self.original is not None else None)


def install(model, root):
    with np.load(root / "weights.npz", allow_pickle=False) as z:
        for i in RAW_MEMBERS:
            expected = f"sbn_fbcnet_raw_k20_s{i - RAW_MEMBERS[0]}.pt"
            if str(z[f"net{i}_sjit_file"]) != expected or not float(z["member_weights"][i + 1]) > 0:
                raise ValueError(f"archived raw member {i} differs")
    adapters = {}
    for _, nets in model.nets.groups:
        for i, net in nets:
            if i in RAW_MEMBERS:
                if type(net).__name__ != "SharedFIRSJit" or net.fblight or i in adapters:
                    raise ValueError("unexpected raw member path")
                adapters[i] = net
    if set(adapters) != set(RAW_MEMBERS) or model.ft.member["fbraw"] != RAW_MEMBERS[0]:
        raise ValueError("missing archived raw member or unexpected fit member")
    return adapters
