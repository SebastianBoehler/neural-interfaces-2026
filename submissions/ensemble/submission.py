"""Tangent-space + ShallowConvNet probability ensemble (Track 2 submission).

Members (all causal, label-free, state reset at every session):
  * AlignedTangent: filter-bank covariances recentred by the causal
    log-Euclidean mean of the session (blended with a training reference as
    ``k`` pseudo-windows), tangent vectors plus a causally centred 1 Hz slow
    waveform, standardised multinomial logistic regression.
  * Shallow: ShallowConvNet (braindecode ShallowFBCSPNet layout, re-implemented
    in plain torch) on windows that are either whitened by the causal running
    mean covariance of the session ("aligned") or divided by a global scale.
The ensemble returns the weighted mean of member probabilities. Parameters
are numpy arrays in ``weights.npz`` (keys prefixed ``ts_`` and ``netI_``).
Per-window work is tiny, so BLAS, OpenMP and torch run single-threaded:
interleaved multi-threaded numpy and torch calls were 4x slower on a loaded
A100 node (spin-waiting thread pools).
"""

import numpy as np
import torch
import torch.nn.functional as F
from scipy.signal import butter, sosfiltfilt

from benchmark_utils.base_solver import CompetSolver

SFREQ = 120.0


def _eig_fn(C, fn):
    w, V = np.linalg.eigh(C)
    return (V * fn(w)) @ V.T


def _log(C):
    return _eig_fn(C, lambda w: np.log(np.maximum(w, 1e-12)))


def _upper(S):
    iu = np.triu_indices(S.shape[-1])
    return S[iu] * np.where(iu[0] == iu[1], 1.0, np.sqrt(2.0))


def _softmax(z):
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


class AlignedTangent:
    def __init__(self, p):
        self.p = p
        self.sos = [butter(4, [lo, hi], btype="band", fs=SFREQ, output="sos")
                    for lo, hi in p["bands"]]
        self.slow_sos = butter(4, float(p["slow_cutoff"]), btype="low",
                               fs=SFREQ, output="sos")
        self.reset_state()

    def reset_state(self):
        self.n = 0
        self.log_sum = [np.zeros_like(g) for g in self.p["log_glob"]]
        self.mean_sum = 0.0

    def _cov(self, x):
        x = x - x.mean(-1, keepdims=True)
        C = (x.astype(np.float64) @ x.T.astype(np.float64)) / x.shape[-1]
        s = float(self.p["shrink"])
        return (1 - s) * C + s * np.trace(C) / len(C) * np.eye(len(C))

    def _features(self, x):
        self.n += 1
        k = float(self.p["k"])
        feats = []
        for b, sos in enumerate(self.sos):
            C = self._cov(sosfiltfilt(sos, x, axis=-1).astype(np.float32))
            L = _log(C)
            self.log_sum[b] += L
            ref = (k * self.p["log_glob"][b] + self.log_sum[b]) / (k + self.n)
            P = _eig_fn(ref, lambda w: np.exp(-0.5 * w))  # expm(ref) ** -1/2
            feats.append(_upper(_log(P @ C @ P)))
        step = int(self.p["slow_step"])
        m = sosfiltfilt(self.slow_sos, x, axis=-1)[..., step // 2::step]
        m = m.reshape(-1).astype(np.float32)
        self.mean_sum = self.mean_sum + m  # float32, like the training cumsum
        feats.append(m - self.mean_sum / (k + self.n))
        return np.concatenate(feats)

    def proba(self, x):
        f = (self._features(x) - self.p["mu"]) / self.p["sd"]
        return _softmax(self.p["W"] @ f + self.p["b"])


class InputStream:
    """Causal per-session network input shared by nets of one alignment."""

    def __init__(self, p):
        self.p = p
        self.reset_state()

    def reset_state(self):
        self.n, self.cov_sum = 0, 0.0

    def __call__(self, x):
        X = x[None].astype(np.float64)
        if not bool(self.p["align"]):
            return (X / float(self.p["scale"])).astype(np.float32)[0]
        xc = x - x.mean(-1, keepdims=True)
        C = np.einsum("nct,ndt->ncd", xc[None], xc[None],
                      dtype=np.float64) / x.shape[-1]
        s = float(self.p["shrink"])
        tr = np.trace(C, axis1=1, axis2=2)[:, None, None] / C.shape[1]
        C = (1 - s) * C + s * tr * np.eye(C.shape[1])
        self.n += 1
        self.cov_sum = self.cov_sum + C[0]
        k = float(self.p["k"])
        R = (k * self.p["G"] + self.cov_sum) / (k + self.n)
        P = _eig_fn(R, lambda w: 1.0 / np.sqrt(np.maximum(w, 1e-12)))
        return np.einsum("ij,njt->nit", P, X).astype(np.float32)[0]


class Shallow:
    """Stateless ShallowConvNet; ``proba`` maps a batch (B, C, T) to (B, 3)."""

    def __init__(self, p):
        t = {k: torch.from_numpy(np.ascontiguousarray(v)) for k, v in p.items()
             if k not in ("align", "k", "G", "scale", "shrink")}
        self.t = t
        # CombinedConv: temporal and spatial filters merged into one kernel
        self.weight = ((t["time_w"] * t["spat_w"].permute(1, 0, 2, 3))
                       .sum(0).unsqueeze(1))
        self.bias = t["spat_w"].squeeze().sum(-1).mm(
            t["time_b"].unsqueeze(-1)).squeeze()

    @torch.no_grad()
    def proba(self, X):
        t = self.t
        h = torch.from_numpy(X).unsqueeze(1).transpose(2, 3)
        h = F.conv2d(h, self.weight, self.bias)            # (B, 40, T', 1)
        h = F.batch_norm(h, t["bn_mean"], t["bn_var"], t["bn_w"], t["bn_b"],
                         False, 0.0, 1e-5)
        h = F.avg_pool2d(h * h, (75, 1), (15, 1))
        h = torch.log(torch.clamp(h, min=1e-6))
        h = F.conv2d(h, t["cls_w"], t["cls_b"])[:, :, 0, 0]
        return torch.softmax(h.double(), 1).numpy()


def _input_key(p):
    keys = ("align", "k", "G", "scale", "shrink")
    return b"".join(np.ascontiguousarray(p[k]).tobytes() for k in keys if k in p)


class Ensemble:
    """Weighted mean of member probabilities.

    Network inputs are computed window by window (causal session state);
    nets are stateless, so they then run once over the whole batch. Nets
    with identical alignment parameters share one input stream.
    """

    def __init__(self, tangent, nets, weights):
        self.tangent, self.w = tangent, weights / weights.sum()
        self.groups = {}
        for i, p in enumerate(nets):
            key = _input_key(p)
            if key not in self.groups:
                self.groups[key] = (InputStream(p), [])
            self.groups[key][1].append((i, Shallow(p)))

    def reset_state(self):
        self.tangent.reset_state()
        for stream, _ in self.groups.values():
            stream.reset_state()

    @torch.no_grad()
    def predict(self, X):
        X = X.detach().cpu().numpy() if torch.is_tensor(X) else np.asarray(X)
        X = X.astype(np.float32)
        inputs = {key: [] for key in self.groups}
        tang = []
        for x in X:  # causal state updates, in acquisition order
            tang.append(self.tangent.proba(x))
            for key, (stream, _) in self.groups.items():
                inputs[key].append(stream(x))
        P = self.w[0] * np.stack(tang)
        for key, (_, nets) in self.groups.items():
            H = np.stack(inputs[key])
            for i, net in nets:
                P = P + self.w[1 + i] * net.proba(H)
        return torch.as_tensor(P.argmax(1), dtype=torch.long)


class Solver(CompetSolver):

    name = "TangentShallowEnsemble"

    def load_model(self, meta):
        torch.set_num_threads(1)
        try:  # bundled with scikit-learn; limits numpy/scipy BLAS + OpenMP
            from threadpoolctl import threadpool_limits
            self._limits = threadpool_limits(limits=1)
        except ImportError:
            pass
        with np.load(meta["submission_dir"] / "weights.npz") as f:
            z = {k: f[k] for k in f.files}
        ts = {k[3:]: v for k, v in z.items() if k.startswith("ts_")}
        ts["log_glob"] = list(ts["log_glob"])
        ts["bands"] = [tuple(b) for b in ts["bands"]]
        nets = []
        for i in range(int(z["n_nets"])):
            pre = f"net{i}_"
            nets.append({k[len(pre):]: v for k, v in z.items() if k.startswith(pre)})
        return Ensemble(AlignedTangent(ts), nets,
                        z["member_weights"].astype(np.float64))
