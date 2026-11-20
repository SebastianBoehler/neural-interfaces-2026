"""Helpers for families/teacher.py: frozen external heads, EA stream and device-free tracing."""

import inspect
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

ARCH_CLASS = {"eegnex": "EEGNeX", "ifnet": "IFNet"}
SHRINK, K = 1e-3, 5.0


def covariances(X, shrink=SHRINK):
    """Trace-shrunk window covariances, identical to experiments/features.py."""
    X = X - X.mean(-1, keepdims=True)
    C = np.einsum("nct,ndt->ncd", X, X, dtype=np.float64) / X.shape[-1]
    tr = np.trace(C, axis1=1, axis2=2)[:, None, None] / C.shape[1]
    return (1 - shrink) * C + shrink * tr * np.eye(C.shape[1])


def ea_params(X):
    """Source-row EA reference: mean window covariance, as AlignedNet._fit for R19e members."""
    G = np.concatenate([covariances(X[i:i + 512]) for i in range(0, len(X), 512)]).mean(0)
    return dict(align=np.bool_(True), k=np.float64(K), G=G,
                scale=np.float64(np.sqrt(np.trace(G) / len(G))), shrink=np.float64(SHRINK))


def ea_stream(X, stream, p):
    """Causal per-session EA exactly as the package InputStream applies it (state reset per stream)."""
    out = np.empty(X.shape, np.float32)
    for i in range(len(X)):
        if i == 0 or stream[i] != stream[i - 1]:
            n, total = 0, 0.0
        x = X[i].astype(np.float32)
        C = covariances(x[None])[0]
        n, total = n + 1, total + C
        w, V = np.linalg.eigh((float(p["k"]) * p["G"] + total) / (float(p["k"]) + n))
        P = (V / np.sqrt(np.maximum(w, 1e-12))) @ V.T
        out[i] = np.einsum("ij,njt->nit", P, x[None].astype(np.float64))[0].astype(np.float32)
    return out


class Heads(nn.Module):
    """Frozen external nets -> (B, families x 19) seed-averaged per-corpus head log-probs."""

    def __init__(self, ckpt_dir, seeds, families):
        super().__init__()
        import braindecode.models as bd
        self.fams = nn.ModuleList()
        self.blocks = []
        for arch in families:
            nets = nn.ModuleList()
            for s in seeds:
                ck = torch.load(Path(ckpt_dir) / f"{arch}_ea-m7r-s{s}.pt", weights_only=False, map_location="cpu")
                corp = json.loads(ck["meta"])["corpora"]
                blocks = [(m["offset"], m["offset"] + len(m["classes"])) for m in corp.values()]
                assert not self.blocks or self.blocks == blocks
                self.blocks = blocks
                cls = getattr(bd, ARCH_CLASS[arch])
                kw = dict(n_chans=41, n_outputs=blocks[-1][1], n_times=480)
                if "sfreq" in inspect.signature(cls).parameters:
                    kw["sfreq"] = 120.0
                net = cls(**kw)
                net.load_state_dict(ck["state"])
                # metadata only: unset braindecode properties raise inside torch.jit.trace's hasattr scan
                net._chs_info = net._chs_info or [{"ch_name": f"ch{c}"} for c in range(41)]
                net._sfreq = net._sfreq or 120.0
                net._input_window_seconds = net._input_window_seconds or 4.0
                nets.append(net.eval())
            self.fams.append(nets)

    def forward(self, x):
        out = []
        for nets in self.fams:
            acc = []
            for net in nets:
                z = net(x)
                acc.append(torch.cat([torch.log_softmax(z[:, a:b], 1) for a, b in self.blocks], 1))
            out.append(torch.stack(acc).mean(0))
        return torch.cat(out, 1)


class Teacher(nn.Module):
    def __init__(self, heads, coef, intercept):
        super().__init__()
        self.heads = heads
        self.lin = nn.Linear(coef.shape[1], 3)
        with torch.no_grad():
            self.lin.weight.copy_(torch.from_numpy(coef).float())
            self.lin.bias.copy_(torch.from_numpy(intercept).float())

    def forward(self, x):
        return self.lin(self.heads(x))


def device_free_trace(module, example):
    """Trace with same-device ``.to(device)`` calls skipped so no CPU constant is baked in."""
    to = torch.Tensor.to

    def same_device_noop(self, *args, **kw):
        if not kw and len(args) == 1 and isinstance(args[0], torch.device) and args[0] == self.device:
            return self
        return to(self, *args, **kw)

    torch.Tensor.to = same_device_noop
    try:
        with torch.no_grad():
            traced = torch.jit.trace(module, example, check_trace=False)
    finally:
        torch.Tensor.to = to
    assert 'value="cpu"' not in traced.inlined_graph.str(), "device constant left in the trace"
    return traced


def batched(f, X, n=256, dev="cpu"):
    with torch.no_grad():
        return torch.cat([f(torch.from_numpy(X[i:i + n]).to(dev)).cpu() for i in range(0, len(X), n)]).double().numpy()
