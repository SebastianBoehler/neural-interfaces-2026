"""s277 shared pieces: paths, channel intersect, euclidean alignment (EA), EEGNet trunk. Own work dir only (ni26_ssl)."""
import hashlib, os, sys
import numpy as np
import torch
from pathlib import Path
WS = Path(os.environ.get("SSL_WS", Path(__file__).resolve().parents[3]))
sys.path[:0] = [str(WS / "experiments")]
import protocol  # noqa: E402  (own copy of experiments/protocol.py)

CORP = ("dreyer2023", "lee2019_mi", "cho2017", "physionetmi", "schirrmeister2017")
EXT = WS / "runs/ext27"; CK = WS / "runs/ckpt"; OUT = WS / "runs/probe"
K_PRIOR, SHRINK, NT, SFREQ = 20.0, 1e-3, 480, 120
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def covs(X):
    """(n, C, T) float tensor -> (n, C, C) float64, demeaned, trace-shrunk as experiments/features.covariances."""
    X = X.double(); X = X - X.mean(-1, keepdim=True)
    C = torch.einsum("nct,ndt->ncd", X, X) / X.shape[-1]
    tr = C.diagonal(dim1=1, dim2=2).sum(-1)[:, None, None] / C.shape[1]
    return (1 - SHRINK) * C + SHRINK * tr * torch.eye(C.shape[1], dtype=C.dtype, device=C.device)


def invsqrtm(R):
    w, V = torch.linalg.eigh(R)
    return (V * w.clamp_min(1e-10).rsqrt()[..., None, :]) @ V.transpose(-1, -2)


def ea_noncausal(X, streams, dev=DEV, bs=512):
    """Per-stream EA over the whole stream. X (n,C,T) numpy float32 -> numpy float16-safe float32 (same order)."""
    out = np.empty(X.shape, np.float32)
    for s in dict.fromkeys(streams):
        rows = np.flatnonzero(streams == s)
        C = torch.cat([covs(torch.from_numpy(X[rows[i:i + bs]]).to(dev)) for i in range(0, len(rows), bs)])
        P = invsqrtm(C.mean(0))
        for i in range(0, len(rows), bs):
            r = rows[i:i + bs]; out[r] = torch.einsum("ij,njt->nit", P, torch.from_numpy(X[r]).to(dev).double()).float().cpu().numpy()
    return out


def ea_causal(X, streams, G, dev=DEV, k=K_PRIOR):
    """Causal EA as AlignedNet._aligned(causal=True): R_n = (k G + cumsum C) / (k + n), prior G (mean training covariance)."""
    out = np.empty(X.shape, np.float32); G = torch.as_tensor(G, device=dev)
    for s in dict.fromkeys(streams):
        rows = np.flatnonzero(streams == s)
        Xt = torch.from_numpy(X[rows]).to(dev); C = covs(Xt)
        n = torch.arange(1, len(rows) + 1, device=dev, dtype=torch.float64)[:, None, None]
        R = (k * G + C.cumsum(0)) / (k + n)
        out[rows] = torch.einsum("nij,njt->nit", invsqrtm(R), Xt.double()).float().cpu().numpy()
    return out


def build_net(n_chans, seed):
    """braindecode EEGNet at its defaults, as experiments/deep.build(eegnet) (n_outputs 3, sfreq 120). Seed fixes the init."""
    import braindecode.models as bd
    torch.manual_seed(seed)
    return bd.EEGNet(n_chans=n_chans, n_outputs=3, n_times=NT, sfreq=SFREQ)


class Trunk(torch.nn.Module):
    """EEGNet up to and including drop_2 (hooked), flattened; BatchNorm only (SessionBN-compatible)."""

    def __init__(self, net):
        super().__init__(); self.net = net; self._h = None
        net.drop_2.register_forward_hook(lambda m, i, o: setattr(self, "_h", o))

    def forward(self, x):
        self.net(x); return self._h.flatten(1)


def trunk_dim(tr, n_chans):
    tr.eval()
    with torch.no_grad():
        return tr(torch.zeros(2, n_chans, NT)).shape[1]


def fold_info(d, i):
    name, tr, te = list(protocol.folds(d, "all_loso"))[i]
    idx = np.flatnonzero(tr); held = str(d["subject"][te][0])
    assert (d["split"][idx] != "external").all() and not (d["subject"][idx] == held).any()
    return name, idx, np.flatnonzero(te), held
