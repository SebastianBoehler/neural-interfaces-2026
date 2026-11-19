"""NI26 Track 2, clean-slate team C: cross-person CNN ensemble with per-session test-time training (TTT).

One (1, 41, 480) window per predict() call; all state is per session (reset_state and __deepcopy__ build it fresh,
the loaded base nets are read-only and shared). Per window:
  1. causal Euclidean alignment: whitening by the shrunk mean covariance of the delivered prefix (current included);
  2. on the adaptation schedule: BN statistics re-estimated on the aligned prefix (mixed with source statistics),
     then gradient steps on the prefix: balanced ensemble pseudo-labels (Sinkhorn) on strong views, information
     maximisation, L2 anchor to the source weights;
  3. output: ensemble mean posterior, optionally rebalanced by a causal prefix Sinkhorn.
No labels, no metadata, nothing crosses a session (module state is constant).
"""
import copy
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from benchmark_utils.base_solver import CompetSolver
except ImportError:  # training / harness outside benchopt
    CompetSolver = object

C_, T_, K_ = 41, 480, 3
DEFAULT_TTT = dict(shrink=0.05, bn="prefix", bn_k=8.0, steps=0, n0=16, every=4, dense_until=48, batch=64, lr=1e-4,
                   params="all", tau=0.0, lam_pl=1.0, lam_im=0.3, anchor=1e-2, temp=1.0, balance=0, bal_from=16,
                   rot=0.1, max_n=1024)


# ----------------------------------------------------------------------------------------------------------- nets
class Shallow(nn.Module):
    def __init__(s, f=40, drop=0.5):
        super().__init__()
        s.t = nn.Conv2d(1, f, (1, 25))
        s.sp = nn.Conv2d(f, f, (C_, 1), bias=False)
        s.bn = nn.BatchNorm2d(f)
        s.pool = nn.AvgPool2d((1, 75), (1, 15))
        s.drop = nn.Dropout(drop)
        s.fc = nn.Linear(f * ((T_ - 24 - 75) // 15 + 1), K_)

    def forward(s, x):
        z = s.bn(s.sp(s.t(x[:, None])))
        z = torch.log(torch.clamp(s.pool(z * z), 1e-6))
        return s.fc(s.drop(z.flatten(1)))


class EEGNet(nn.Module):
    def __init__(s, f1=16, d=2, f2=32, k=60, drop=0.4):
        super().__init__()
        s.b1 = nn.Sequential(nn.Conv2d(1, f1, (1, k), padding=(0, k // 2), bias=False), nn.BatchNorm2d(f1),
                             nn.Conv2d(f1, f1 * d, (C_, 1), groups=f1, bias=False), nn.BatchNorm2d(f1 * d), nn.ELU(),
                             nn.AvgPool2d((1, 4)), nn.Dropout(drop))
        s.b2 = nn.Sequential(nn.Conv2d(f1 * d, f1 * d, (1, 16), padding=(0, 8), groups=f1 * d, bias=False),
                             nn.Conv2d(f1 * d, f2, 1, bias=False), nn.BatchNorm2d(f2), nn.ELU(),
                             nn.AvgPool2d((1, 8)), nn.Dropout(drop))
        s.fc = nn.Linear(f2 * (((T_ + 1) // 4 + 1) // 8), K_)

    def forward(s, x):
        return s.fc(s.b2(s.b1(x[:, None])).flatten(1))


def _bandpass_bank(bands, n=61, fs=120.0):
    t = np.arange(n) - n // 2
    win = np.hamming(n)
    out = []
    for lo, hi in bands:
        h = (2 * hi / fs) * np.sinc(2 * hi * t / fs) - (2 * lo / fs) * np.sinc(2 * lo * t / fs)
        out.append(h * win)
    return torch.tensor(np.array(out), dtype=torch.float32)


class FBNet(nn.Module):
    """Learnable filter bank (FIR-initialised) -> per-band spatial filters -> log-variance over segments."""
    BANDS = ((1, 4), (4, 8), (8, 12), (12, 16), (16, 20), (20, 24), (24, 28), (28, 32), (32, 40))

    def __init__(s, m=4, seg=4, drop=0.3):
        super().__init__()
        nb = len(s.BANDS)
        s.fb = nn.Conv2d(1, nb, (1, 61), padding=(0, 30), bias=False)
        s.fb.weight.data[:, 0, 0] = _bandpass_bank(s.BANDS)
        s.sp = nn.Conv2d(nb, nb * m, (C_, 1), groups=nb, bias=False)
        s.bn = nn.BatchNorm2d(nb * m)
        s.seg, s.drop = seg, nn.Dropout(drop)
        s.fc = nn.Linear(nb * m * seg, K_)

    def forward(s, x):
        z = s.bn(s.sp(s.fb(x[:, None])))[:, :, 0]
        z = z.reshape(z.shape[0], z.shape[1], s.seg, -1)
        z = torch.log(torch.clamp(z.var(-1), 1e-6))
        return s.fc(s.drop(z.flatten(1)))


ARCHS = dict(shallow=Shallow, eegnet=EEGNet, fbnet=FBNet)


def build(arch, kw=None):
    return ARCHS[arch](**(kw or {}))


# ------------------------------------------------------------------------------------------- alignment, augmentation
def inv_sqrt(C, shrink):
    """Batched (..., C, C) float64 -> float32 inverse square root of the shrunk covariance."""
    tr = torch.diagonal(C, dim1=-2, dim2=-1).mean(-1)[..., None, None]
    C = (1 - shrink) * C + shrink * tr * torch.eye(C.shape[-1], dtype=C.dtype, device=C.device)
    w, V = torch.linalg.eigh(C)
    return ((V * w.clamp_min(1e-12).rsqrt()[..., None, :]) @ V.transpose(-1, -2)).float()


def augment(x, g, rot=0.1, shift=24, noise=0.1, drop=0.1, scale=0.15):
    """Strong view of aligned windows (B, C, T): rotation near identity (virtual subject), circular time shift,
    amplitude scaling, channel dropout, white noise. g: torch.Generator on x's device."""
    B, C, T = x.shape
    dev = x.device
    if rot > 0:
        A = torch.randn(B, C, C, generator=g, device=dev) * (rot / math.sqrt(C))
        x = torch.linalg.matrix_exp(A - A.transpose(1, 2)) @ x
    if shift > 0:
        s = torch.randint(-shift, shift + 1, (B, 1, 1), generator=g, device=dev)
        x = torch.gather(x, 2, ((torch.arange(T, device=dev)[None, None] + s) % T).expand(B, C, T))
    x = x * torch.exp(scale * torch.randn(B, 1, 1, generator=g, device=dev))
    x = x * (torch.rand(B, C, 1, generator=g, device=dev) > drop)
    return x + noise * torch.randn(x.shape, generator=g, device=dev)


def sinkhorn(P, iters=10):
    """Rows of P (n, K) rescaled so the column sums are equal (balanced soft assignment), rows renormalised."""
    Q = P.clamp_min(1e-8)
    for _ in range(iters):
        Q = Q / Q.sum(0, keepdim=True)
        Q = Q / Q.sum(1, keepdim=True)
    return Q


# ------------------------------------------------------------------------------------------------- streaming model
class TTTModel:
    def __init__(self, nets, cfg, dev):
        self.base = [n.to(dev).eval() for n in nets]
        self.cfg = {**DEFAULT_TTT, **(cfg or {})}
        unknown = set(self.cfg) - set(DEFAULT_TTT)
        if unknown:
            raise KeyError(f"unknown TTT keys {sorted(unknown)}")
        self.dev = dev
        self.reset_state()

    def __deepcopy__(self, memo):  # base nets are shared and never modified; session state is rebuilt
        new = object.__new__(TTTModel)
        new.base, new.cfg, new.dev = self.base, self.cfg, self.dev
        new.reset_state()
        return new

    def reset_state(self):
        c, dev = self.cfg, self.dev
        self.nets = [copy.deepcopy(n) for n in self.base]
        self.X = torch.zeros(c["max_n"], C_, T_, device=dev)
        self.covsum = torch.zeros(C_, C_, dtype=torch.float64, device=dev)
        self.n = 0
        self.P = torch.zeros(c["max_n"], K_, device=dev)
        self.g = torch.Generator(device=dev)
        self.g.manual_seed(1234)
        self.opts = None
        self.last = None

    # -- helpers
    def _bn_refresh(self, A):
        k = self.cfg["bn_k"]
        n = A.shape[0]
        for net, base in zip(self.nets, self.base):
            bns = [m for m in net.modules() if isinstance(m, nn.BatchNorm2d)]
            for m in bns:
                m.reset_running_stats()
                m.momentum = None
                m.train()
            with torch.no_grad():
                net(A)
            for m, b in zip(bns, [m for m in base.modules() if isinstance(m, nn.BatchNorm2d)]):
                w = n / (n + k)
                m.running_mean.mul_(w).add_((1 - w) * b.running_mean)
                m.running_var.mul_(w).add_((1 - w) * b.running_var)
                m.eval()

    def _params(self, net):
        if self.cfg["params"] == "all":
            return list(net.parameters())
        ps = [p for m in net.modules() if isinstance(m, nn.BatchNorm2d) for p in m.parameters()]
        if self.cfg["params"] == "bn+fc":
            ps += list(net.fc.parameters())
        return ps

    def _probs(self, A):
        with torch.no_grad():
            return torch.stack([F.softmax(net(A) / self.cfg["temp"], -1) for net in self.nets]).mean(0)

    def _steps(self, A):
        c = self.cfg
        if self.opts is None:
            self.opts = [torch.optim.Adam(self._params(n), lr=c["lr"]) for n in self.nets]
            self.anchor = [[p.detach().clone() for p in self._params(n)] for n in self.nets]
        n = A.shape[0]
        Q = sinkhorn(self._probs(A))
        conf = (Q.max(1).values >= c["tau"]).float()
        for _ in range(c["steps"]):
            idx = torch.randint(0, n, (min(c["batch"], n),), generator=self.g, device=self.dev)
            xs = augment(A[idx], self.g, rot=c["rot"])
            q, w = Q[idx], conf[idx]
            for net, opt, anc in zip(self.nets, self.opts, self.anchor):
                logp = F.log_softmax(net(xs), -1)
                p = logp.exp()
                loss = c["lam_pl"] * (-(q * logp).sum(1) * w).sum() / w.sum().clamp_min(1)
                if c["lam_im"]:
                    pm = p.mean(0)
                    loss = loss + c["lam_im"] * (-(p * logp).sum(1).mean() + (pm * pm.clamp_min(1e-8).log()).sum())
                if c["anchor"]:
                    loss = loss + c["anchor"] * sum(((a - b) ** 2).sum() for a, b in zip(self._params(net), anc))
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()

    def _due(self):
        c, n = self.cfg, self.n
        return n >= c["n0"] and (n <= c["dense_until"] or n % c["every"] == 0)

    # -- official entry point
    def predict(self, X):
        c = self.cfg
        x = torch.as_tensor(X, dtype=torch.float32, device=self.dev).reshape(1, C_, T_)
        if self.n >= c["max_n"]:  # buffer full: keep the most recent half (never expected for ~200-window sessions)
            h = c["max_n"] // 2
            self.X[:h] = self.X[-h:].clone()
            self.P[:h] = self.P[-h:].clone()
            self.n = h
        x64 = x[0].double()
        self.covsum += x64 @ x64.T / T_
        self.X[self.n] = x[0]
        self.n += 1
        W = inv_sqrt(self.covsum / self.n, c["shrink"])
        A = W @ self.X[: self.n]
        if self._due():
            if c["bn"] == "prefix":
                self._bn_refresh(A)
            if c["steps"] > 0:
                self._steps(A)
        p = self._probs(A[-1:])[0]
        self.P[self.n - 1] = p
        if c["balance"] and self.n >= c["bal_from"]:
            p = sinkhorn(self.P[: self.n])[-1]
        self.last = p
        return torch.argmax(p).reshape(1).cpu()


def load_package(path, dev, cfg=None):
    blob = torch.load(Path(path) / "weights.pt", map_location=dev, weights_only=False)
    nets = []
    for m in blob["members"]:
        net = build(m["arch"], m.get("kw"))
        net.load_state_dict(m["state"])
        nets.append(net)
    return TTTModel(nets, {**blob["ttt"], **(cfg or {})}, dev)


class Solver(CompetSolver):
    name = "teamC"

    def load_model(self, meta):
        torch.backends.cudnn.benchmark = False
        return load_package(meta["submission_dir"], meta.get("device", "cpu"))
