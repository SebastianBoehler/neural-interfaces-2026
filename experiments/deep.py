"""Deep candidates on Euclidean-aligned windows (same harness contract).

Training sessions are whitened by their mean covariance. At test time window
t is whitened by the running mean covariance of windows 0..t of its session,
blended with the global training mean as ``k`` pseudo-windows.
"""

import numpy as np
import torch
from braindecode.models import EEGNet, ShallowFBCSPNet
from torch import nn

from features import covariances, invsqrtm

ARCHS = {"eegnet": EEGNet, "shallow": ShallowFBCSPNet}


def device():
    if torch.cuda.is_available():
        return "cuda"
    return "mps" if torch.backends.mps.is_available() else "cpu"


class AlignedNet:
    _covs = {}

    def __init__(self, d, arch="eegnet", align=True, k=5.0, epochs=40,
                 lr=1e-3, wd=1e-3, seed=0, batch=64):
        self.d, self.arch, self.align, self.k = d, arch, align, k
        self.epochs, self.lr, self.wd, self.batch = epochs, lr, wd, batch
        self.seed, self.dev = seed, device()
        key = id(d["X"])
        if key not in self._covs:
            self._covs[key] = np.concatenate(
                [covariances(d["X"][i:i + 512]) for i in range(0, len(d["X"]), 512)])
        self.C = self._covs[key]

    def _aligned(self, idx, causal):
        X = self.d["X"][idx].astype(np.float64)
        if not self.align:
            return (X / self.scale).astype(np.float32)
        if causal:
            n = np.arange(1, len(idx) + 1)[:, None, None]
            R = (self.k * self.G + np.cumsum(self.C[idx], 0)) / (self.k + n)
            P = invsqrtm(R)
            return np.einsum("nij,njt->nit", P, X).astype(np.float32)
        P = invsqrtm(self.C[idx].mean(0))
        return np.einsum("ij,njt->nit", P, X).astype(np.float32)

    def fit(self, idx):
        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)
        self.G = self.C[idx].mean(0)
        self.scale = float(np.sqrt(np.trace(self.G) / len(self.G)))
        streams = self.d["stream"][idx]
        parts = [(self._aligned(idx[streams == s], False), self.d["y"][idx[streams == s]])
                 for s in dict.fromkeys(streams)]
        X = torch.from_numpy(np.concatenate([p[0] for p in parts]))
        y = torch.from_numpy(np.concatenate([p[1] for p in parts])).long()
        n_ch, n_t = X.shape[1:]
        self.net = ARCHS[self.arch](n_chans=n_ch, n_outputs=3, n_times=n_t).to(self.dev)
        opt = torch.optim.AdamW(self.net.parameters(), lr=self.lr, weight_decay=self.wd)
        steps = self.epochs * int(np.ceil(len(X) / self.batch))
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, self.lr, total_steps=steps)
        loss_fn = nn.CrossEntropyLoss()
        self.net.train()
        for _ in range(self.epochs):
            perm = rng.permutation(len(X))
            for b in range(0, len(X), self.batch):
                j = perm[b:b + self.batch]
                xb, yb = X[j].to(self.dev), y[j].to(self.dev)
                opt.zero_grad()
                loss_fn(self.net(xb), yb).backward()
                opt.step()
                sched.step()
        self.net.eval()

    @torch.no_grad()
    def predict_proba_stream(self, idx):
        X = torch.from_numpy(self._aligned(idx, causal=True)).to(self.dev)
        return torch.softmax(self.net(X).cpu().double(), 1).numpy()

    def predict_stream(self, idx):
        return self.predict_proba_stream(idx).argmax(1)
