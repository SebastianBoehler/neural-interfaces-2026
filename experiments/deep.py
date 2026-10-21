"""Deep candidates on Euclidean-aligned windows (same harness contract).

Training sessions are whitened by their mean covariance. At test time window
t is whitened by the running mean covariance of windows 0..t of its session,
blended with the global training mean as ``k`` pseudo-windows.

``adabn`` > 0 also adapts every batch-norm layer causally at test time
(``adabn`` = 0: no adaptation, but trained nets are shared across configs):
window t is normalised with the per-channel mean and variance of windows
0..t of its session, blended with the training running statistics as
``adabn`` pseudo-windows (no labels, reset per session).
"""

import copy
import re
import inspect
import os

import braindecode.models as bd
import numpy as np
import torch
from torch import nn

from features import SFREQ, covariances, invsqrtm

CROPS = {}  # start (s after cue) -> training crops, rows as the official export


def crops(start, d):
    """Windows starting ``start`` s after the cue (scripts/export_windows.py
    START), checked to match the official export row by row."""
    if start not in CROPS:
        from protocol import DATA
        z = np.load(DATA.with_name(f"proteus_windows_start{start:g}.npz"))
        n = len(z["y"])
        for k in ("y", "subject", "session", "run"):
            assert (z[k] == d[k][:n]).all(), k
        CROPS[start] = z["X"]
    return CROPS[start]

ARCHS = {"eegnet": "EEGNet", "shallow": "ShallowFBCSPNet", "deep4": "Deep4Net",
         "conformer": "EEGConformer", "atcnet": "ATCNet", "eegnex": "EEGNeX",
         "ctnet": "CTNet", "fbcnet": "FBCNet", "ifnet": "IFNet", "tidnet": "TIDNet",
         "cbramod": "CBraMod", "labram": "Labram", "reve": "REVE"}
# Foundation models: Hub weights (BSD-3-Clause) at a pinned revision, input
# rate, and amplitude factor: our inputs have unit RMS and CBraMod was
# pretrained on 100-uV units, so 0.15 matches PROTEUS' ~15 uV median RMS.
REVE_REV = "dc2a075c287bb2f6c04ee5875bd79535a0f7dba6"
PRETRAINED = {"cbramod": ("braindecode/cbramod-pretrained",
                          "584cdc415913739a05d84bf0c1cb3db397764507", 200, 0.15),
              "labram": ("braindecode/labram-pretrained",  # BSD-3-Clause, 0.1 mV
                         "0563b6c626e7b40d9a36653b763715db94d945d7", 200, 0.15)}


class Resampled(nn.Module):
    """Linear resampling to the pretrained model's rate, then scaling."""

    def __init__(self, net, n_times, factor, clamp=None, **fwd):
        super().__init__()
        self.net, self.n_times, self.factor, self.fwd = net, n_times, factor, fwd
        self.clamp = clamp

    def forward(self, x):
        x = nn.functional.interpolate(x, size=self.n_times, mode="linear",
                                      align_corners=False) * self.factor
        if self.clamp:
            x = x.clamp(-self.clamp, self.clamp)
        return self.net(x, **self.fwd)


def build(arch, n_chans, n_times, pretrained=True, **net_kw):
    if arch == "reve":  # REVE responsible-use licence v1.0; staged on workers.
        from protocol import DATA  # Aligned input is ~unit variance, like
        names = [str(c) for c in np.load(DATA)["ch_names"]]  # REVE's z-scores
        n = int(round(n_times * 200 / SFREQ))
        kw = dict(n_outputs=3, n_chans=n_chans, n_times=n, sfreq=200,
                  chs_info=[{"ch_name": c} for c in names])
        net = (bd.REVE.from_pretrained("brain-bzh/reve-base", revision=REVE_REV, **kw)
               if pretrained else bd.REVE(**kw))
        assert net.default_pos is not None
        return Resampled(net, n, 1.0, clamp=15.0)
    if arch in PRETRAINED:
        repo, rev, sfreq, factor = PRETRAINED[arch]
        n = int(round(n_times * sfreq / SFREQ))
        kw = dict(n_outputs=3, n_chans=n_chans, n_times=n, sfreq=sfreq)
        cls = getattr(bd, ARCHS[arch])
        net = cls(**kw)
        if pretrained:  # fail loudly instead of a silently random backbone
            from huggingface_hub import hf_hub_download
            from safetensors.torch import load_file
            sd = load_file(hf_hub_download(repo, "model.safetensors", revision=rev))
            sd = {k: v for k, v in sd.items() if not k.startswith("final_layer.")}
            if "temporal_embedding" in sd:  # LaBraM: CLS + our n // 200 patches
                sd["temporal_embedding"] = sd["temporal_embedding"][:, :n // 200 + 1]
            res = net.load_state_dict(sd, strict=False)
            assert not res.unexpected_keys, res.unexpected_keys
            assert all(k.startswith("final_layer.") for k in res.missing_keys), res
        fwd = {}
        if arch == "labram":  # position embeddings selected by channel name
            from protocol import DATA
            fwd["ch_names"] = [str(c) for c in np.load(DATA)["ch_names"]]
        return Resampled(net, n, factor, **fwd)
    cls = getattr(bd, ARCHS[arch])
    kw = dict(n_chans=n_chans, n_outputs=3, n_times=n_times)
    if "sfreq" in inspect.signature(cls).parameters:
        kw["sfreq"] = SFREQ
    return cls(**kw, **net_kw)  # architecture variants, e.g. FBCNet stride


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


def causal_bn(bn, x, k):
    """Batch norm of one session's windows (rows in order) with causal
    statistics: training running stats as ``k`` pseudo-windows + rows 0..t."""
    red = list(range(2, x.dim()))
    m, s = (x.mean(red), (x * x).mean(red)) if red else (x, x * x)
    n = torch.arange(1, len(x) + 1, device=x.device, dtype=x.dtype)[:, None]
    rm, rv = bn.running_mean, bn.running_var
    mean = (k * rm + m.cumsum(0)) / (k + n)
    var = ((k * (rv + rm ** 2) + s.cumsum(0)) / (k + n) - mean ** 2).clamp_min(0)
    shape = (len(x), -1) + (1,) * len(red)
    out = (x - mean.view(shape)) / torch.sqrt(var.view(shape) + bn.eps)
    if bn.affine:
        out = out * bn.weight.view(shape[1:2] + shape[2:]) + bn.bias.view(shape[1:2] + shape[2:])
    return out


def _depth(name):
    """Transformer block index of a parameter (CBraMod ``layers``, LaBraM
    ``blocks``), -1 for embeddings, None for the head."""
    m = re.search(r"\.(?:layers|blocks)\.(\d+)\.", name)
    if m:
        return int(m.group(1))
    if re.search(r"patch_embed|position_embedding|temporal_embedding|cls_token|"
                 r"to_patch_embedding|mlp4d|fourier4d|net\.ln\.", name):
        return -1
    return None


def param_groups(net, lr, lld, head_only=False):
    """AdamW groups: with layer-wise decay ``lld`` < 1 transformer block i of
    L gets lr * lld ** (L - i), the embeddings lr * lld ** (L + 1) and the
    head lr; ``head_only`` trains just the final layer (linear probe)."""
    if lld == 1 and not head_only:  # default: one group, as before
        for prm in net.parameters():
            prm.requires_grad_(True)
        return [{"params": list(net.parameters()), "lr": lr}]
    named = list(net.named_parameters())
    L = len({_depth(n) for n, _ in named} - {None, -1})
    groups = {}
    for n, prm in named:
        if head_only and "final_layer" not in n:
            prm.requires_grad_(False)
            continue
        prm.requires_grad_(True)
        depth = _depth(n)
        groups.setdefault(L if depth is None else depth, []).append(prm)
    return [{"params": v, "lr": lr * lld ** (L - k)} for k, v in groups.items()]


class AlignedNet:
    _covs = {}
    _fitted = {}  # trained nets by data, settings and training rows (reused
                  # by configs that differ only in test-time adaptation)

    def __init__(self, d, arch="eegnet", align=True, k=5.0, epochs=40,
                 lr=1e-3, wd=1e-3, seed=0, batch=64, aug=False, ft=0, adabn=None,
                 pretrained=True, ls=0.0, lld=1.0, lp=0, crops=(), net_kw=None):
        self.d, self.arch, self.align, self.k, self.ft = d, arch, align, k, ft
        self.adabn, self.pretrained = adabn, pretrained
        self.ls, self.lld, self.lp = ls, lld, lp  # fine-tuning recipe
        self.crops = tuple(crops)  # extra training windows (s after the cue)
        self.net_kw = dict(net_kw or {})
        self.epochs, self.lr, self.wd, self.batch = epochs, lr, wd, batch
        self.seed, self.dev, self.aug = seed, device(), aug
        key = id(d["X"])
        if key not in self._covs:
            self._covs[key] = np.concatenate(
                [covariances(d["X"][i:i + 512]) for i in range(0, len(d["X"]), 512)])
        self.C = self._covs[key]

    def _aligned(self, idx, causal, src=None):
        """``src``: other windows of the same rows (crops), whitened with
        the official windows' statistics."""
        X = (self.d["X"] if src is None else src)[idx].astype(np.float64)
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
        key = (id(self.d["X"]), self.arch, self.align, self.k, self.epochs, self.lr,
               self.wd, self.seed, self.batch, self.aug, self.ft, self.dev,
               self.pretrained, self.ls, self.lld, self.lp, self.crops,
               tuple(sorted(self.net_kw.items())),
               hash(np.asarray(idx).tobytes()))
        if self.adabn is None:  # only the *_bn* configs reuse trained nets
            return self._fit(idx)
        if key not in self._fitted:  # kept on CPU to bound device memory
            self._fit(idx)
            self._fitted[key] = (copy.deepcopy(self.net).cpu(), self.G, self.scale)
        net, self.G, self.scale = self._fitted[key]
        self.net = copy.deepcopy(net).to(self.dev)

    def _fit(self, idx):
        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)
        self.G = self.C[idx].mean(0)
        self.scale = float(np.sqrt(np.trace(self.G) / len(self.G)))
        streams = self.d["stream"][idx]
        order = np.concatenate([idx[streams == s] for s in dict.fromkeys(streams)])
        X = [self._aligned(idx[streams == s], False) for s in dict.fromkeys(streams)]
        for start in self.crops:  # same rows and labels, shifted windows
            src = crops(start, self.d)
            assert order.max() < len(src), "crops exist for PROTEUS rows only"
            X += [self._aligned(idx[streams == s], False, src)
                  for s in dict.fromkeys(streams)]
        order = np.tile(order, 1 + len(self.crops))
        X = torch.from_numpy(np.concatenate(X))
        y = torch.from_numpy(self.d["y"][order]).long()
        sw = torch.from_numpy(self.d["w"][order]).float() if "w" in self.d else None
        n_ch, n_t = X.shape[1:]
        self.net = build(self.arch, n_ch, n_t, self.pretrained,
                         **self.net_kw).to(self.dev)
        weighted = bool(self.d.get("tag"))  # external data: class-balanced,
        if weighted:                          # per-sample weighted loss
            cnt = np.bincount(y.numpy(), weights=sw.numpy(), minlength=3)
            cw = torch.tensor(sw.sum().item() / (3 * cnt), dtype=torch.float32)
            sw = (sw * cw[y]).to(self.dev)
        loss_fn = nn.CrossEntropyLoss(reduction="none" if weighted else "mean",
                                      label_smoothing=self.ls)
        gen = torch.Generator().manual_seed(self.seed)
        self.net.train()
        rows = np.arange(len(X))
        if self.lp:  # linear probe first, at 10x the learning rate
            self._train(X, y, sw, rows, self.lp, 10 * self.lr, rng, gen, loss_fn,
                        weighted, head_only=True)
        self._train(X, y, sw, rows, self.epochs, self.lr, rng, gen, loss_fn, weighted)
        if self.ft and weighted:  # fine-tune on PROTEUS windows only
            rows = np.flatnonzero(self.d["split"][order] != "external")
            self._train(X, y, sw, rows, self.ft, self.lr / 3, rng, gen, loss_fn, weighted)
        self.net.eval()

    def _train(self, X, y, sw, rows, epochs, lr, rng, gen, loss_fn, weighted,
               head_only=False):
        groups = param_groups(self.net, lr, self.lld, head_only)
        opt = torch.optim.AdamW(groups, lr=lr, weight_decay=self.wd)
        steps = epochs * int(np.ceil(len(rows) / self.batch))
        sched = torch.optim.lr_scheduler.OneCycleLR(
            opt, [g["lr"] for g in groups], total_steps=steps)
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
        bns = [m for m in self.net.modules()
               if self.adabn and isinstance(m, nn.modules.batchnorm._BatchNorm)]
        for bn in bns:
            bn.forward = lambda x, bn=bn: causal_bn(bn, x, self.adabn)
        try:
            logits = self.net(X)
        finally:
            for bn in bns:
                del bn.forward
        return torch.softmax(logits.cpu().double(), 1).numpy()

    def predict_stream(self, idx):
        return self.predict_proba_stream(idx).argmax(1)
