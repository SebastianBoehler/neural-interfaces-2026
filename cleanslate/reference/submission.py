"""Clean reference system: causal EA + compact EEGNet + causal adaptive BatchNorm (+ optional class-balance prior).

No stage chain, no routing, no personalisation. Components (switches live in config.json, written by train.py from env
CS_REF_EA / CS_REF_ADABN / CS_REF_PRIOR, default 1/1/0; the Solver reads ONLY config.json so a package cannot disagree
with its description):
  ea     : whiten each window by the mean covariance of this and all earlier windows of the session (shrunk to identity
           over the first N0 windows). Off = raw window.
  adabn  : eval-time BatchNorm statistics = (K * train running stats + n * prefix-window stats) / (K + n), prefix = the
           delivered windows of this session including the current one. Off = frozen train statistics.
  prior  : logits - ALPHA * log(running mean of the session's softmax outputs) (pushes predictions toward balance).
Session state is held on the model object and cleared by reset_state(); nothing is module-level.
"""
import json

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from benchmark_utils.base_solver import CompetSolver

N0, K_BN, ALPHA = 10.0, 20.0, 0.5


def _isqrt(R):
    w, V = np.linalg.eigh(R)
    return (V * (1 / np.sqrt(np.maximum(w, 1e-10)))) @ V.T


class CausalEA:
    def __init__(self, d=41):
        self.d = d
        self.reset()

    def reset(self):
        self.S, self.n = np.zeros((self.d, self.d)), 0

    def apply(self, x):
        x = x.astype(np.float64)
        x = x - x.mean(1, keepdims=True)
        self.S += x @ x.T / x.shape[1]
        self.n += 1
        R = self.S / self.n
        R = (self.n * R + N0 * np.trace(R) / self.d * np.eye(self.d)) / (self.n + N0)
        return (_isqrt(R) @ x).astype(np.float32)


class AdaBN(nn.BatchNorm2d):
    """BatchNorm2d whose eval statistics can follow the delivered prefix (per-window stats averaged over windows)."""
    adapt = False

    def reset_state(self):
        self.n, self.sm, self.sv = 0, 0.0, 0.0

    def forward(self, x):
        if self.training or not self.adapt:
            return super().forward(x)
        m, v = x.mean((0, 2, 3)), x.var((0, 2, 3), unbiased=False)
        self.n += 1
        self.sm, self.sv = self.sm + m, self.sv + v
        w = self.n / (self.n + K_BN)
        mu = (1 - w) * self.running_mean + w * self.sm / self.n
        var = (1 - w) * self.running_var + w * self.sv / self.n
        y = (x - mu[None, :, None, None]) / torch.sqrt(var + self.eps)[None, :, None, None]
        return y * self.weight[None, :, None, None] + self.bias[None, :, None, None]


class EEGNet(nn.Module):
    def __init__(self, n_chans=41, n_times=480, n_classes=3, f1=8, d=2, f2=16):
        super().__init__()
        self.t = nn.Conv2d(1, f1, (1, 64), padding=(0, 32), bias=False)
        self.b1 = AdaBN(f1)
        self.s = nn.Conv2d(f1, f1 * d, (n_chans, 1), groups=f1, bias=False)
        self.b2 = AdaBN(f1 * d)
        self.sep = nn.Conv2d(f1 * d, f1 * d, (1, 16), padding=(0, 8), groups=f1 * d, bias=False)
        self.pw = nn.Conv2d(f1 * d, f2, 1, bias=False)
        self.b3 = AdaBN(f2)
        self.drop = nn.Dropout(0.5)
        self.fc = nn.Linear(f2 * (n_times // 32), n_classes)

    def forward(self, x):
        x = self.b1(self.t(x[:, None]))
        x = self.drop(F.avg_pool2d(F.elu(self.b2(self.s(x))), (1, 4)))
        x = self.drop(F.avg_pool2d(F.elu(self.b3(self.pw(self.sep(x)))), (1, 8)))
        return self.fc(x.flatten(1))


class _Model:
    def __init__(self, net, cfg, device):
        self.net, self.cfg, self.device = net, cfg, device
        for m in net.modules():
            if isinstance(m, AdaBN):
                m.adapt = bool(cfg["adabn"])
        self.reset_state()

    def reset_state(self):
        self.ea = CausalEA()
        self.psum, self.pn = np.zeros(3), 0
        for m in self.net.modules():
            if isinstance(m, AdaBN):
                m.reset_state()

    @torch.no_grad()
    def predict(self, X):
        out = []
        for x in X.detach().cpu().numpy():
            x = self.ea.apply(x) if self.cfg["ea"] else x.astype(np.float32)
            logit = self.net(torch.from_numpy(x)[None].to(self.device))[0].double().cpu().numpy()
            p = np.exp(logit - logit.max())
            p /= p.sum()
            self.psum, self.pn = self.psum + p, self.pn + 1
            if self.cfg["prior"]:
                logit = logit - ALPHA * np.log(self.psum / self.pn + 1e-3)
            out.append(int(np.argmax(logit)))
        return torch.tensor(out, dtype=torch.int64)


class Solver(CompetSolver):
    name = "cs_reference"

    def load_model(self, meta):
        d = meta["submission_dir"]
        cfg = json.load(open(f"{d}/config.json"))
        net = EEGNet()
        net.load_state_dict(torch.load(f"{d}/weights.pt", map_location="cpu"))
        return _Model(net.to(meta["device"]).eval(), cfg, meta["device"])
