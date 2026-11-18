"""One-window F16e S1C with strict-past bounded SessionBN."""
import copy

import torch
from torch import nn
from torch.nn.utils.parametrize import is_parametrized, remove_parametrizations


class BoundedSession(nn.Module):
    def __init__(self, prior):
        super().__init__()
        if (prior.k, prior.m, prior.causal) != (20., 32, False) or prior.weight is None:
            raise ValueError("expected the exact F16e S1C normalization prior")
        self.k = float(prior.k)
        self.eps = float(prior.eps)
        for name, tensor in (("rm", prior.running_mean), ("rv", prior.running_var),
                             ("affine_weight", prior.weight), ("affine_bias", prior.bias)):
            self.register_buffer(name, tensor.detach().clone())
        self.register_buffer("hist_n", torch.zeros((), dtype=torch.int64))
        self.register_buffer("hist_s1", torch.zeros_like(self.rm, dtype=torch.float64))
        self.register_buffer("hist_s2", torch.zeros_like(self.rm, dtype=torch.float64))
        self.register_buffer("hist_mass", torch.zeros_like(self.hist_s1))
        self.register_buffer("hist_w1", torch.zeros_like(self.hist_s1))
        self.register_buffer("hist_w2", torch.zeros_like(self.hist_s1))
        self.register_buffer("hist_clipped", torch.zeros_like(self.rm, dtype=torch.bool))

    @torch.jit.export
    def reset(self) -> torch.Tensor:
        self.hist_n.zero_()
        self.hist_s1.zero_()
        self.hist_s2.zero_()
        self.hist_mass.zero_()
        self.hist_w1.zero_()
        self.hist_w2.zero_()
        self.hist_clipped.zero_()
        return self.hist_n.clone()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 4 or x.size(0) != 1:
            raise ValueError("bounded actor accepts one activation window")
        r1 = x.mean((2, 3))[0]
        r2 = (x * x).mean((2, 3))[0]
        s1 = self.hist_s1 + r1.double()
        s2 = self.hist_s2 + r2.double()
        n = self.hist_n + 1
        total = self.k + n.to(dtype=x.dtype)
        mean = (self.k * self.rm + s1.to(dtype=x.dtype)) / total
        var = ((self.k * (self.rv + self.rm.square()) + s2.to(dtype=x.dtype)) / total - mean.square()).clamp_min(0)

        x64 = x.double()
        m1 = x64.mean((2, 3))[0]
        m2 = x64.square().mean((2, 3))[0]
        rm64, rv64 = self.rm.double(), self.rv.double()
        mass = self.k + 1 + self.hist_mass
        center = (self.k * rm64 + m1 + self.hist_w1) / mass
        spread = ((self.k * (rv64 + rm64.square()) + m2 + self.hist_w2) / mass - center.square()).clamp_min(0)
        mean = torch.where(self.hist_clipped, center.to(dtype=x.dtype), mean)
        var = torch.where(self.hist_clipped, spread.to(dtype=x.dtype), var)
        out = (x - mean.view(1, -1, 1, 1)) / torch.sqrt(var.view(1, -1, 1, 1) + self.eps)
        out = out * self.affine_weight.view(1, -1, 1, 1) + self.affine_bias.view(1, -1, 1, 1)

        energy = m2 - 2 * m1 * rm64 + rm64.square()
        new_weight = (4 * rv64.clamp_min(self.eps) / energy.clamp_min(self.eps)).clamp(max=1.0)
        self.hist_n.copy_(n)
        self.hist_s1.copy_(s1)
        self.hist_s2.copy_(s2)
        self.hist_mass.add_(new_weight)
        self.hist_w1.add_(new_weight * m1)
        self.hist_w2.add_(new_weight * m2)
        self.hist_clipped.logical_or_(new_weight < 1)
        return out


class ExportedActor(nn.Module):
    def __init__(self, trained, normalizer, scale):
        super().__init__()
        if trained.body.spatial_conv[1] is not normalizer or len(trained.body.spatial_conv) != 3:
            raise ValueError("original FBCNet spatial topology differs")
        self.body = copy.deepcopy(trained.body).cpu().eval()
        self.task = copy.deepcopy(trained.task).cpu().eval()
        # Braindecode's trace discovers these metadata properties.
        self.body._chs_info = self.body._chs_info or [{"ch_name": f"ch{c}"} for c in range(41)]
        self.body._sfreq = self.body._sfreq or 120.
        self.body._input_window_seconds = self.body._input_window_seconds or 4.
        conv = self.body.spatial_conv[0]
        if not is_parametrized(conv, "weight"):
            raise ValueError("original S1C convolution lost its fitted MaxNorm")
        effective = conv.weight.detach().clone()
        remove_parametrizations(conv, "weight", leave_parametrized=True)
        if not torch.equal(conv.weight, effective):
            raise ValueError("frozen spatial weight differs from its native MaxNorm value")
        self.body.spatial_conv[1] = torch.jit.script(BoundedSession(normalizer).eval())
        self.register_buffer("scale", torch.tensor(scale, dtype=torch.float32))
        self.register_buffer("fixed_zero", torch.zeros((1, 287), dtype=torch.float32))

    def forward(self, window: torch.Tensor) -> torch.Tensor:
        waveform = self.body(window / self.scale).flatten(1)
        logits = self.task(torch.cat((waveform, self.fixed_zero), dim=1))
        return torch.softmax(logits.double(), dim=1)

    def reset(self) -> torch.Tensor:
        return self.body.spatial_conv[1].reset()


@torch.no_grad()
def export_actor(trained, normalizer, scale, sample):
    actor = ExportedActor(trained, normalizer, scale).eval()
    actor.reset()
    compiled = torch.jit.trace_module(actor, {"forward": (sample,), "reset": ()}, check_trace=False)
    if compiled.reset().item() != 0:
        raise ValueError("exported reset did not clear the history")
    return compiled
