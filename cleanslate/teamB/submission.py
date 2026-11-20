"""Clean-slate team B: cluster-then-name session model on a clusterability-trained encoder ensemble.

Per session (state resets with the session, nothing crosses sessions):
1. Causal Euclidean alignment: whiten by the mean 1-45 Hz covariance of the delivered prefix (current window included).
2. Encoders (several families x seeds) embed the window; every K-th window the whole prefix is re-embedded under the
   current alignment. Session feature normalisation (SFN) standardises each embedding over the prefix.
3. Each encoder gives class logits (cross-person "naming") and a unit projection trained with a within-session
   supervised-contrastive loss ("clusterability": same-class windows of one session lie together).
4. Transduction: von Mises-Fisher mixture on the projections of the prefix, initialised and named by the ensemble
   posterior, with a class-balance prior that tightens as the prefix grows. The current window's responsibility is
   the answer. Arms (`enc`, `bal`, `em`) are kept per window so the adaptation gain is measured apart from the encoder.
   Label propagation (`lp`, off unless lp_beta > 0): leave-one-out kernel ridge on the prefix's tangent-space vectors
   (9 bands, aligned covariances) spreads the soft labels along the session's own geometry; generic features, nothing learned.
5. Returning people (optional, when the package holds person experts): at fixed prefix lengths every public person's
   expert (the encoders fine-tuned on that person) labels the prefix; the expert whose predictions are most confident and
   balanced (information-maximisation score) is mixed in if it stands out from the other experts (z-score gate). No
   identity is given; a new person should leave the gate closed.
Env switches (read once in load_model, printed): CSB_FINAL=enc|bal|em, CSB_KAPPA, CSB_ALPHA, CSB_NB, CSB_K, CSB_GATE_Z, CSB_XW, CSB_LP_BETA, CSB_LP_LAM.
"""
import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from benchmark_utils.base_solver import CompetSolver
except ImportError:  # train.py imports this file outside the official tree
    CompetSolver = object

FS, T, C, PD = 120, 480, 41, 32
BANDS = ((1, 4), (4, 8), (8, 12), (12, 16), (16, 20), (20, 24), (24, 28), (28, 36), (36, 45))
DEFAULTS = dict(final="em", kappa=8.0, alpha=1.0, nb=24.0, K=4, iters=6, n0=8, sfn_k=4.0, gate_z=2.5, xw=0.5,
                gate_at=(16, 32, 64, 128), lp_beta=0.0, lp_lam=1.0, lp_n0=12, lp_rounds=2,
                ts_at=(24, 32, 48, 64, 96, 128, 192))


def fmask(lo, hi, edge=0.5):
    f = torch.fft.rfftfreq(T, 1 / FS)
    return ((f - lo) / edge + 1).clamp(0, 1) * ((hi - f) / edge + 1).clamp(0, 1)


def bandpass(x, mask):
    """x (B,C,T); mask (F,) -> (B,C,T) or (K,F) -> (B,K,C,T). Zero-phase FFT filter inside the window."""
    X = torch.fft.rfft(x)
    if mask.dim() == 1:
        return torch.fft.irfft(X * mask, n=T)
    return torch.fft.irfft(X[:, None] * mask[None, :, None, :], n=T)


def detrend(x):
    t = torch.linspace(-1, 1, T, device=x.device, dtype=x.dtype)
    x = x - x.mean(-1, keepdim=True)
    return x - (x @ t)[..., None] / (t @ t) * t


def whitener(S, n):
    """S: sum of n window covariances (..,C,C) float64 -> R^-1/2 with one pseudo-window of scaled identity."""
    eye = torch.eye(C, dtype=S.dtype, device=S.device)
    R = (S + S.diagonal(dim1=-2, dim2=-1).sum(-1)[..., None, None] / (n * C) * eye) / (n + 1)
    w, V = torch.linalg.eigh(R)
    return (V * w.clamp_min(1e-12).rsqrt()[..., None, :]) @ V.transpose(-1, -2)


def sfn(e, k):
    """Standardise e (..,n,D) over its n rows, with k pseudo-rows of the unit prior."""
    n = e.shape[-2]
    mu = e.sum(-2, keepdim=True) / (n + k)
    var = (((e - mu) ** 2).sum(-2, keepdim=True) + k) / (n + k)
    return (e - mu) * var.rsqrt()


class Shallow(nn.Module):
    def __init__(self, lo=1.0, hi=45.0, kern=25, nf=40):
        super().__init__()
        self.register_buffer("mask", fmask(lo, hi))
        self.tc = nn.Conv2d(1, nf, (1, kern), padding=(0, kern // 2))
        self.sc = nn.Conv2d(nf, nf, (C, 1), bias=False)
        self.bn = nn.BatchNorm2d(nf)
        self.pool = nn.AvgPool2d((1, 75), (1, 15))
        self.dim = nf * ((T - 75) // 15 + 1)

    def forward(self, x):
        h = self.bn(self.sc(self.tc(bandpass(x, self.mask)[:, None])))
        return torch.log(self.pool(h * h).clamp_min(1e-6)).flatten(1)


class FBC(nn.Module):
    def __init__(self, m=8, seg=4):
        super().__init__()
        self.register_buffer("mask", torch.stack([fmask(lo, hi) for lo, hi in BANDS]))
        nb = len(BANDS)
        self.sc = nn.Conv2d(nb, nb * m, (C, 1), groups=nb, bias=False)
        self.bn = nn.BatchNorm2d(nb * m)
        self.seg, self.dim = seg, nb * m * seg

    def forward(self, x):
        h = self.bn(self.sc(bandpass(x, self.mask)))
        v = h.view(h.shape[0], h.shape[1], self.seg, -1).var(-1)
        return torch.log(v.clamp_min(1e-6)).flatten(1)


BODIES = dict(shallow=lambda: Shallow(), slow=lambda: Shallow(0.3, 30.0, 49), fbc=lambda: FBC())


class Net(nn.Module):
    def __init__(self, kind):
        super().__init__()
        self.body = BODIES[kind]()
        d = self.body.dim
        self.bn, self.drop, self.head = nn.BatchNorm1d(d), nn.Dropout(0.5), nn.Linear(d, 3)
        self.proj = nn.Sequential(nn.Linear(d, 128), nn.ReLU(), nn.Linear(128, PD))

    def feats(self, x):
        return self.bn(self.body(x))

    def heads(self, z):
        return self.head(self.drop(z)), F.normalize(self.proj(z), dim=-1)


def balance(logr, rho):
    """Class-prior shift of strength rho in [0,1] toward equal class mass over the rows."""
    b = torch.zeros(3, dtype=logr.dtype, device=logr.device)
    for _ in range(10):
        b = b - torch.log(3 * torch.softmax(logr + b, 1).mean(0).clamp_min(1e-6))
    return torch.softmax(logr + rho * b, 1)


def transduce(P, U, cfg):
    """P (n,3) ensemble posterior, U (n,D) unit projections -> (balanced P, mixture responsibilities)."""
    n = P.shape[0]
    logP, rho = torch.log(P + 1e-6), n / (n + cfg["nb"])
    r = bal = balance(logP, rho)
    if n < cfg["n0"]:
        return bal, bal
    for _ in range(cfg["iters"]):
        mu = F.normalize(r.T @ U, dim=1)
        r = balance(cfg["alpha"] * logP + cfg["kappa"] * rho * (U @ mu.T), rho)
    return bal, r


def tangent(xs, bank):
    """Aligned windows (m,C,T) -> log-Euclidean tangent vectors of the band covariances, (m, bands*C*(C+1)/2)."""
    out, iu = [], torch.triu_indices(C, C, device=xs.device)
    scale = torch.where(iu[0] == iu[1], 1.0, math.sqrt(2.0)).to(xs.device)
    eye = torch.eye(C, dtype=torch.float64, device=xs.device)
    for i in range(0, len(xs), 32):
        xb = bandpass(xs[i:i + 32], bank).double()
        cov = xb @ xb.transpose(-1, -2) / T
        cov = cov + 1e-3 * cov.diagonal(dim1=-2, dim2=-1).mean(-1)[..., None, None] * eye
        w, V = torch.linalg.eigh(cov)
        L = (V * torch.log(w.clamp_min(1e-12))[..., None, :]) @ V.transpose(-1, -2)
        out.append((L[..., iu[0], iu[1]] * scale).flatten(1).float())
    return torch.cat(out)


class TSM(nn.Module):
    """Tangent-space family: per-band linear reduction of the aligned covariance's matrix logarithm."""

    def __init__(self, k=24):
        super().__init__()
        self.register_buffer("mask", torch.stack([fmask(lo, hi) for lo, hi in BANDS]))
        self.red = nn.Parameter(torch.randn(len(BANDS), C * (C + 1) // 2, k) * 0.03)
        self.dim = len(BANDS) * k

    def forward(self, x):
        with torch.no_grad():
            v = tangent(x, self.mask).view(len(x), len(BANDS), -1)
        return torch.einsum("bnd,ndk->bnk", v, self.red).flatten(1)


BODIES["tsm"] = lambda: TSM()


def propagate(P, V, cfg):
    """Leave-one-out kernel-ridge label propagation of the soft labels P (n,3) over tangent vectors V (n,D)."""
    n = P.shape[0]
    if cfg["lp_beta"] <= 0 or n < cfg["lp_n0"]:
        return P
    Z = V - V.mean(0)
    Z = Z / Z.norm(dim=1).mean().clamp_min(1e-9)
    Kmat = (Z @ Z.T).double()
    H = Kmat @ torch.linalg.inv(Kmat + cfg["lp_lam"] * torch.eye(n, dtype=torch.float64, device=P.device))
    h = H.diagonal()[:, None].clamp(max=0.95)
    logP, Q = torch.log(P + 1e-6), P
    for _ in range(cfg["lp_rounds"]):
        R = (balance(torch.log(Q + 1e-6), n / (n + cfg["nb"])) - 1 / 3).double()
        Q = torch.softmax(logP + cfg["lp_beta"] * ((H @ R - h * R) / (1 - h)).float(), 1)
    return Q


class Model:
    def __init__(self, pack, device):
        self.dev, self.cfg = device, dict(DEFAULTS, **pack.get("cfg", {}))
        for k in ("final", "kappa", "alpha", "nb", "K", "gate_z", "xw", "lp_beta", "lp_lam"):
            v = os.environ.get("CSB_" + k.upper())
            if v is not None:
                self.cfg[k] = type(DEFAULTS[k])(v)
                print(f"[teamB] env override {k}={self.cfg[k]}", flush=True)
        assert self.cfg["final"] in ("enc", "exp", "lp", "bal", "em"), self.cfg["final"]
        def build(items):
            nets = []
            for kind, state in items:
                net = Net(kind)
                net.load_state_dict(state)
                nets.append(net.to(device).eval())
            return nets
        self.nets = build(pack["nets"])
        assert self.nets, "package holds no encoder"
        self.experts = {p: build(items) for p, items in pack.get("experts", {}).items()}
        self.mask = fmask(1.0, 45.0).to(device)
        self.bank = torch.stack([fmask(lo, hi) for lo, hi in BANDS]).to(device)
        self.reset_state()

    def reset_state(self):
        self.X, self.S, self.E, self.last = [], None, None, None
        self.xp, self.XE, self.gate_log, self.V = None, None, [], None
        self.arms = dict(enc=[], exp=[], lp=[], bal=[], em=[])

    def _embed(self, nets, xs):
        return [torch.cat([net.feats(xs[i:i + 64]) for i in range(0, len(xs), 64)]) for net in nets]

    def _logp(self, nets, embs):
        heads = [net.heads(sfn(E, self.cfg["sfn_k"])) for net, E in zip(nets, embs)]
        return sum(torch.log_softmax(lg, 1) for lg, _ in heads) / len(nets), [q for _, q in heads]

    def _gate(self, xs):
        """Pick the person expert whose prefix predictions are most confident and balanced, if it stands out."""
        people, scores = sorted(self.experts), []
        for p in people:
            P = torch.softmax(self._logp(self.experts[p], self._embed(self.experts[p], xs))[0], 1)
            ent = lambda q: -(q * torch.log(q + 1e-9)).sum(-1)
            scores.append(float(ent(P.mean(0)) - ent(P).mean()))
        s = torch.tensor(scores)
        best = int(s.argmax())
        rest = torch.cat([s[:best], s[best + 1:]])
        z = float((s[best] - rest.mean()) / rest.std().clamp_min(1e-6)) if len(rest) > 1 else 0.0
        self.gate_log.append((len(xs), people[best], z, scores))
        self.xp = people[best] if z > self.cfg["gate_z"] else None

    @torch.no_grad()
    def predict(self, X):
        cfg = self.cfg
        x = detrend(X[0].to(self.dev, torch.float32))
        self.X.append(x)
        n = len(self.X)
        xb = bandpass(x[None], self.mask)[0].double()
        c = xb @ xb.T / T
        self.S = c if self.S is None else self.S + c
        W = whitener(self.S, n).float()
        gate = bool(self.experts) and n in cfg["gate_at"]
        if cfg["lp_beta"] > 0:  # tangent vectors: whole prefix at a few lengths, the current window otherwise
            if n <= 16 or n in cfg["ts_at"]:
                self.V = tangent(W @ torch.stack(self.X), self.bank)
            else:
                self.V = torch.cat([self.V, tangent((W @ x)[None], self.bank)])
        if n <= 16 or n % cfg["K"] == 0 or gate:
            xs = W @ torch.stack(self.X)
            self.E = self._embed(self.nets, xs)
            if gate:
                self._gate(xs)
            self.XE = self._embed(self.experts[self.xp], xs) if self.xp else None
        else:
            xa = (W @ x)[None]
            self.E = [torch.cat([E, net.feats(xa)]) for net, E in zip(self.nets, self.E)]
            if self.xp:
                self.XE = [torch.cat([E, net.feats(xa)]) for net, E in zip(self.experts[self.xp], self.XE)]
        logp, U = self._logp(self.nets, self.E)
        enc = int(logp[-1].argmax())
        if self.xp:
            logp = (1 - cfg["xw"]) * logp + cfg["xw"] * self._logp(self.experts[self.xp], self.XE)[0]
        P = torch.softmax(logp, 1)
        exp = int(P[-1].argmax())
        P = propagate(P, self.V, cfg)
        U = torch.cat(U, 1) / math.sqrt(len(self.nets))
        bal, r = transduce(P, U, cfg)
        self.last = (P, U, r)
        out = dict(enc=enc, exp=exp, lp=int(P[-1].argmax()), bal=int(bal[-1].argmax()), em=int(r[-1].argmax()))
        for k, v in out.items():
            self.arms[k].append(v)
        return torch.tensor([out[cfg["final"]]], dtype=torch.int64)


class Solver(CompetSolver):
    name = "cs_teamB"

    def load_model(self, meta):
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        dev = str(meta.get("device", "cpu"))
        pack = torch.load(f"{meta['submission_dir']}/weights.pt", map_location=dev, weights_only=True)
        return Model(pack, dev)
