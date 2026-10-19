"""Deep candidates on Euclidean-aligned windows (same harness contract).

Training sessions are whitened by their mean covariance. At test time window
t is whitened by the running mean covariance of windows 0..t of its session,
blended with the global training mean as ``k`` pseudo-windows.
"""

import inspect
import os

import braindecode.models as bd
import numpy as np
import torch
from torch import nn

from features import SFREQ, covariances, invsqrtm

ARCHS = {"eegnet": "EEGNet", "shallow": "ShallowFBCSPNet", "deep4": "Deep4Net",
         "conformer": "EEGConformer", "atcnet": "ATCNet", "eegnex": "EEGNeX",
         "ctnet": "CTNet", "fbcnet": "FBCNet", "ifnet": "IFNet", "tidnet": "TIDNet"}


def build(arch, n_chans, n_times):
    cls = getattr(bd, ARCHS[arch])
    kw = dict(n_chans=n_chans, n_outputs=3, n_times=n_times)
    if "sfreq" in inspect.signature(cls).parameters:
        kw["sfreq"] = SFREQ
    return cls(**kw)


def device():
    if os.environ.get("NI26_DEVICE"):
        return os.environ["NI26_DEVICE"]
    if torch.cuda.is_available():
        return "cuda"
    return "mps" if torch.backends.mps.is_available() else "cpu"


def augment(x, gen, shift=30, scale=0.1, drop=0.1):
    """Training-time augmentation: circular time shift (<= ``shift``
    samples), per-window amplitude scaling and channel dropout."""
    B, C, _ = x.shape
    k = int(torch.randint(-shift, shift + 1, (1,), generator=gen))
    x = torch.roll(x, k, dims=-1)
    amp = 1 + scale * (2 * torch.rand(B, 1, 1, generator=gen) - 1)
    keep = (torch.rand(B, C, 1, generator=gen) >= drop).float()
    return x * (amp * keep).to(x.device)


class AlignedNet:
    _covs = {}

    def __init__(self, d, arch="eegnet", align=True, k=5.0, epochs=40,
                 lr=1e-3, wd=1e-3, seed=0, batch=64, aug=False, ft=0):
        self.d, self.arch, self.align, self.k, self.ft = d, arch, align, k, ft
        self.epochs, self.lr, self.wd, self.batch = epochs, lr, wd, batch
        self.seed, self.dev, self.aug = seed, device(), aug
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
        order = np.concatenate([idx[streams == s] for s in dict.fromkeys(streams)])
        X = torch.from_numpy(np.concatenate(
            [self._aligned(idx[streams == s], False) for s in dict.fromkeys(streams)]))
        y = torch.from_numpy(self.d["y"][order]).long()
        sw = torch.from_numpy(self.d["w"][order]).float() if "w" in self.d else None
        n_ch, n_t = X.shape[1:]
        self.net = build(self.arch, n_ch, n_t).to(self.dev)
        weighted = bool(self.d.get("tag"))  # external data: class-balanced,
        if weighted:                          # per-sample weighted loss
            cnt = np.bincount(y.numpy(), weights=sw.numpy(), minlength=3)
            cw = torch.tensor(sw.sum().item() / (3 * cnt), dtype=torch.float32)
            sw = (sw * cw[y]).to(self.dev)
        loss_fn = nn.CrossEntropyLoss(reduction="none" if weighted else "mean")
        gen = torch.Generator().manual_seed(self.seed)
        self.net.train()
        rows = np.arange(len(X))
        self._train(X, y, sw, rows, self.epochs, self.lr, rng, gen, loss_fn, weighted)
        if self.ft and weighted:  # fine-tune on PROTEUS windows only
            rows = np.flatnonzero(self.d["split"][order] != "external")
            self._train(X, y, sw, rows, self.ft, self.lr / 3, rng, gen, loss_fn, weighted)
        self.net.eval()

    def _train(self, X, y, sw, rows, epochs, lr, rng, gen, loss_fn, weighted):
        opt = torch.optim.AdamW(self.net.parameters(), lr=lr, weight_decay=self.wd)
        steps = epochs * int(np.ceil(len(rows) / self.batch))
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, lr, total_steps=steps)
        for _ in range(epochs):
            perm = rows[rng.permutation(len(rows))]
            for b in range(0, len(perm), self.batch):
                j = perm[b:b + self.batch]
                xb, yb = X[j].to(self.dev), y[j].to(self.dev)
                if self.aug:
                    xb = augment(xb, gen)
                opt.zero_grad()
                loss = loss_fn(self.net(xb), yb)
                if weighted:
                    loss = (loss * sw[j]).sum() / sw[j].sum()
                loss.backward()
                opt.step()
                sched.step()

    @torch.no_grad()
    def predict_proba_stream(self, idx):
        X = torch.from_numpy(self._aligned(idx, causal=True)).to(self.dev)
        return torch.softmax(self.net(X).cpu().double(), 1).numpy()

    def predict_stream(self, idx):
        return self.predict_proba_stream(idx).argmax(1)
