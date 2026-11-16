"""s288: TSMNet (Kobler et al. 2022, simplified) as a SessionBN-family member.

spatio-temporal conv -> BatchNorm (-> SessionBN via SessionNet) -> covariance over time (SPD) -> domain-specific SPD-BN
-> BiMap -> ReEig -> LogEig -> linear. All SPD algebra in float64.

SPD-BN (SPDSessionBN): recentres each session by the log-Euclidean Frechet mean M = exp(mean log S): S -> M^-1/2 S M^-1/2
(congruence, so a session's mean becomes the identity). Train: each chunk of m rows is ONE session (SessionNet sampler);
its own mean is used. Eval: windows of one session in stream order; window t uses (k prior + sum_{u<=t} log S_u) / (k + t + 1),
the prior being the EMA of the training chunks' mean logs (zeros = identity at the start): causal, label-free, current session only.
The recentring matrix is not differentiated (statistics are constants in the backward pass). Simplifications against the paper,
stated: log-Euclidean instead of affine-invariant Frechet mean, no learned bias / dispersion parameter.
"""
import numpy as np
import torch
from torch import nn

import deep
import session_bn as SB

D64 = torch.float64


def sym_fn(S, f):
    """f applied to the eigenvalues of a batch of symmetric matrices (float64)."""
    w, U = torch.linalg.eigh(S)
    return (U * f(w).unsqueeze(-2)) @ U.transpose(-1, -2)


def logm(S):
    return sym_fn(S, torch.log)


class SPDSessionBN(nn.Module):
    def __init__(self, d, momentum=0.1):
        super().__init__()
        self.d, self.momentum = d, momentum
        self.register_buffer("run_log", torch.zeros(d, d, dtype=D64))  # EMA of chunk mean logs: the population prior (identity at start)
        self.m, self.k = 0, 20.0  # chunk size (train), prior pseudo-windows (eval); set by TSMSessionNet
        self.steps = 0

    def recentre(self, S, Lbar):
        """S (n, d, d), Lbar (n, d, d) mean log per row -> M^-1/2 S M^-1/2 with M = exp(Lbar) (Lbar detached)."""
        with torch.no_grad():
            P = sym_fn(Lbar, lambda w: torch.exp(-0.5 * w))
        return P @ S @ P

    def forward(self, S):
        with torch.no_grad():
            L = logm(S.detach())
        if self.training:
            m = self.m or len(S)
            G = len(S) // m
            assert G * m == len(S), (len(S), m)  # chunks are single sessions (SessionNet._batches)
            Lc = L.reshape(G, m, self.d, self.d).mean(1)
            with torch.no_grad():
                self.run_log.mul_(1 - self.momentum).add_(self.momentum * Lc.mean(0)); self.steps += 1
            Lbar = Lc.repeat_interleave(m, 0)
        else:
            n = torch.arange(1, len(S) + 1, device=S.device, dtype=D64)[:, None, None]
            Lbar = (self.k * self.run_log + L.cumsum(0)) / (self.k + n)
        return self.recentre(S, Lbar)


class BiMap(nn.Module):
    """W^T S W with W (d_in x d_out) kept on the Stiefel manifold by QR of a free parameter."""

    def __init__(self, d_in, d_out):
        super().__init__()
        self.A = nn.Parameter(torch.linalg.qr(torch.randn(d_in, d_out))[0])

    def forward(self, S):
        W = torch.linalg.qr(self.A.to(D64))[0]
        return W.T @ S @ W


class TSMNet(nn.Module):
    def __init__(self, n_chans=41, n_times=480, n_outputs=3, f_t=4, f_s=40, d_out=20, k_t=25, eps=1e-4):
        super().__init__()
        self.eps = eps
        self.conv_t = nn.Conv2d(1, f_t, (1, k_t), padding=(0, k_t // 2), bias=False)
        self.conv_s = nn.Conv2d(f_t, f_s, (n_chans, 1), bias=False)
        self.bn = nn.BatchNorm2d(f_s)  # becomes SessionBN in SessionNet._prepare
        self.spdbn = SPDSessionBN(f_s)
        self.bimap = BiMap(f_s, d_out)
        self.register_buffer("iu", torch.triu_indices(d_out, d_out), persistent=False)
        self.final_layer = nn.Linear(self.iu.shape[1], n_outputs)

    def spd_features(self, x):
        z = self.bn(self.conv_s(self.conv_t(x.unsqueeze(1)))).squeeze(2).to(D64)  # (B, f_s, T)
        S = z @ z.transpose(1, 2) / z.shape[-1]
        return S + self.eps * torch.eye(S.shape[-1], dtype=D64, device=S.device)

    def forward(self, x):
        S = self.bimap(self.spdbn(self.spd_features(x)))
        S = sym_fn(S, lambda w: w.clamp_min(self.eps))  # ReEig
        Z = logm(S)  # LogEig
        v = Z[:, self.iu[0], self.iu[1]] * torch.where(self.iu[0] == self.iu[1], 1.0, 2 ** 0.5).to(Z.device, D64)
        return self.final_layer(v.float())


class TSMSessionNet(SB.SessionNet):
    """SessionNet whose net is a TSMNet: its BatchNorm becomes SessionBN, its SPD-BN takes the same chunk size m and test prior k."""

    def _prepare(self):
        super()._prepare()
        for mod in self.spdbns():
            mod.m = self.group

    def spdbns(self):
        return [m for m in self.net.modules() if isinstance(m, SPDSessionBN)]

    def set_k(self, k):
        super().set_k(k)
        for mod in self.spdbns():
            mod.k = k


_build = deep.build


def build(arch, n_chans, n_times, pretrained=True, **kw):
    return TSMNet(n_chans, n_times, **kw) if arch == "tsmnet" else _build(arch, n_chans, n_times, pretrained, **kw)


def make(d, seed):
    """the s288 family 'tsm': TSMNet, EA input, rot05, SessionBN + SPD-BN, same recipe as the other members (no pretrained trunk)."""
    deep.build = build  # AlignedNet._fit looks build up in deep's globals
    deep.ARCHS.setdefault("tsmnet", "TSMNet")
    net = TSMSessionNet(d, arch="tsmnet", seed=seed, pretrained=False, **SB.CFGS["sbn"]); net.rot = 0.05
    return net


def check_tsm(m, d, te, proto="all_loso"):
    """guard: SPD-BN active (statistics updated in training, k set, one layer), held-out person not among the training people,
    the SPD-BN output differs from its input (it recentres)."""
    held = set(np.unique(d["subject"][te]).tolist()); sb = m.spdbns()
    assert len(sb) == 1 and sb[0].steps > 0 and float(sb[0].run_log.abs().max()) > 0, "SPD-BN not active"
    assert sb[0].m == m.group and sb[0].k == m.sbns()[0].k, (sb[0].m, sb[0].k)
    if proto == "all_loso":
        assert not (held & m.fit_subjects), f"held-out person(s) {held & m.fit_subjects} among the training people"
    print(f"tsm ok: SPD-BN steps {sb[0].steps}, |run_log|max {float(sb[0].run_log.abs().max()):.3f}, k {sb[0].k}, "
          f"{len(m.fit_subjects)} training people, held-out {sorted(held)}", flush=True)
