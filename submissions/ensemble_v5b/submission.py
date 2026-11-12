"""Tangent-space + network probability ensemble (Track 2 submission; docs/STATE_7NOV.md).

Lean code of 7 Nov: it keeps only the paths of the three uploaded sealed packages (cand_nxif,
cand_rec_ts2_tsem, cand_rec_ts2_tsem_route3geb), which it reproduces byte for byte; other staged
packages (routing v1 / v2, plain EB, CBraMod members) are rebuilt with commit c4d600d.

Members (causal, label-free, state reset at every session), weighted by ``member_weights``:
  * AlignedTangent (``ts_``; optional second member ``ts2_``, last weight): filter-bank
    covariances recentred by the session's causal log-Euclidean mean (blended with a training
    prior as ``k`` pseudo-windows), tangent vectors plus a causally centred 1 Hz slow waveform,
    standardised multinomial LR. ``car`` / ``nsub`` / ``hilbert``: the 6 Nov cores.
  * Nets (``netI_``): ShallowConvNet, FBCNet, FBLightConvNet, SincShallowNet (plain torch) and
    TorchScript members (``jit_file``; session-stateful ``sjit_file``) on windows whitened by the
    session's causal mean covariance ("align") or divided by a global scale.
Stages after the mix, each causal per session: rebalancing (``rb_*``: divide by the mean of the
last ``W`` mixes ** alpha), SessionEM (``em_*``: Gaussian class templates in a frontal
slow-potential + ocular-event view), TSEM (``tsem_*``: the same in a PCA view of the first
tangent member's scaled features), optional gated early boost (``em_boost``, ``tsem_boost``,
``eb_*``) and returning-person routing v3 (``route_*``: a gate on that unrouted output feeds the
matched public person's head into the core1 slot of a second, routed rebalancing -> SessionEM
-> TSEM tail). Per-window work is tiny, so BLAS, OpenMP and torch run single-threaded:
interleaved multi-threaded numpy and torch calls were 4x slower on a loaded A100 node.

Routing v4 (11 Nov, optional ``bio_*`` keys): a label-free identity gate (class Bio: spectral-shape session
signature against the public people's galleries, from window ``bio_t0``) whose ramp is taken when it exceeds the v3
gate's; without the keys the outputs equal the fast package's.
v5.1b (12 Nov evening): identical outputs to v5.1; Route3 writes one stderr line per session about the own-space stage.
Own-space stage (v5, 12 Nov, optional ``own_*`` keys; class OwnStage): on windows routed to public person j with
ramp r, the routed output is multiplied by exp(own_beta * r * mean clipped log posterior of three personal members
fitted in that person's own space); without the keys the outputs equal the v4.1 package's.
Runtime version (11 Nov, cand_rec_ts2_tsem_route3geb_fast; same outputs as the lean code, only the
member mix is computed faster; every stage after the mix is the lean code unchanged):
  * nets with member weight 0 are not loaded or run, and the device-side member posteriors are
    copied to the host once per window, then summed in the original order on the CPU;
  * the 24 fingerprinted FBCNet / FBLightConvNet SessionBN traces of one input group share one
    float32 FIR filter-bank output (``SharedFIRSJit``; other traces use the generic SJit);
  * the tangent members (CPU) run on one worker thread while the network members run on the
    main thread, for the same window; all session state still advances before predict returns.
"""

import copy
import hashlib
import os
import sys
from concurrent.futures import ThreadPoolExecutor, wait
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.signal import butter, hilbert, sosfilt, sosfilt_zi, sosfiltfilt

from benchmark_utils.base_solver import CompetSolver

SFREQ = 120.0
# Training montage order (PROTEUS meta["ch_names"]); see channel_map
CH_NAMES = ("C2", "CP2", "P2", "Pz", "CPz", "Cz", "C1", "CP1", "P1", "P3", "PO7", "P5", "TP7",
            "CP5", "CP3", "C3", "PO8", "P6", "P4", "TP8", "CP6", "CP4", "C6", "C4", "FC6", "FC4",
            "FC2", "F6", "F4", "F2", "AF4", "AF3", "F5", "F3", "F1", "Fz", "FC5", "FC3", "FC1",
            "FCz", "C5")


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


def _eig_b(C, fn):  # batched _eig_fn, (B, n, n); torch's eigh is faster here; complex Hermitian too
    w, V = (t.numpy() for t in torch.linalg.eigh(torch.from_numpy(C)))
    return (V * fn(w)[:, None, :]) @ (V.conj() if np.iscomplexobj(V) else V).transpose(0, 2, 1)


def _filtfilt(sos, zi, n, x):
    """scipy.signal.sosfiltfilt(sos, x, axis=-1) for 2-D x (odd padding of n =
    its default padlen, zi = sosfilt_zi(sos)): the same steps without the
    per-call setup."""
    e = np.concatenate([2 * x[:, :1] - x[:, n:0:-1], x, 2 * x[:, -1:] - x[:, -2:-(n + 2):-1]], -1)
    y = sosfilt(sos, e, zi=zi[:, None] * e[:, :1])[0]
    y = sosfilt(sos, y[:, ::-1], zi=zi[:, None] * y[:, -1:])[0]
    return y[:, ::-1][:, n:-n]


class AlignedTangent:
    def __init__(self, p):
        self.p = p
        self.sos = [butter(4, [lo, hi], btype="band", fs=SFREQ, output="sos")
                    for lo, hi in p["bands"]]
        self.slow_sos = butter(4, float(p["slow_cutoff"]), btype="low",
                               fs=SFREQ, output="sos")
        # Optional (6 Nov core, default off): ts_car = common-average reference of
        # the window; ts_nsub = per band the full window plus nsub sub-window
        # covariances (blocks band-major), vectorised, float64 slow block;
        # ts_hilbert (nsub path) = covariances of the bands' analytic signals (complex
        # Hermitian; tangent vector per block = Re upper + sqrt(2) Im strict upper)
        self.car = bool(p.get("car", False))
        self.nsub = int(p.get("nsub", 0))
        self.hilbert = bool(p.get("hilbert", False))
        self.glob = np.stack(p["log_glob"]) if self.nsub else None
        self.filt = [(s, sosfilt_zi(s), 3 * (2 * len(s) + 1 - min((s[:, 2] == 0).sum(), (s[:, 5] == 0).sum())))
                     for s in self.sos] if self.nsub else None
        self.reset_state()

    def reset_state(self):
        self.n = 0
        self.log_sum = [np.zeros_like(g) for g in self.p["log_glob"]]
        if self.nsub:
            self.log_sum = np.zeros_like(self.glob)
        self.mean_sum = 0.0

    def _features_sub(self, x):
        self.n += 1
        k, s = float(self.p["k"]), float(self.p["shrink"])
        B = np.stack([_filtfilt(*f, x) for f in self.filt]).astype(np.float32)
        if self.hilbert:
            B = hilbert(B, axis=-1).astype(np.complex64)
        C = []
        for seg in [B] + np.array_split(B, self.nsub, axis=-1):
            seg = (seg - seg.mean(-1, keepdims=True)).astype(np.complex128 if self.hilbert else np.float64)
            S = seg @ (seg.conj() if self.hilbert else seg).transpose(0, 2, 1) / seg.shape[-1]
            tr = np.trace(S, axis1=1, axis2=2)[:, None, None].real / S.shape[1]
            C.append((1 - s) * S + s * tr * np.eye(S.shape[1]))
        C = np.stack(C, 1).reshape(-1, *C[0].shape[1:])
        self.log_sum += _eig_b(C, lambda w: np.log(np.maximum(w, 1e-12)))
        ref = (k * self.glob + self.log_sum) / (k + self.n)
        P = _eig_b(ref, lambda w: np.exp(-0.5 * w))
        T = _eig_b(P @ C @ P, lambda w: np.log(np.maximum(w, 1e-12)))
        iu = np.triu_indices(T.shape[-1])
        V = T.real[:, iu[0], iu[1]] * np.where(iu[0] == iu[1], 1.0, np.sqrt(2.0))
        if self.hilbert:
            ju = np.triu_indices(T.shape[-1], 1)
            V = np.concatenate([V, np.sqrt(2.0) * T.imag[:, ju[0], ju[1]]], 1)
        feats = [V.reshape(-1)]
        step = int(self.p["slow_step"])
        m = sosfiltfilt(self.slow_sos, x, axis=-1)[..., step // 2::step]
        m = m.reshape(-1).astype(np.float32).astype(np.float64)
        self.mean_sum = self.mean_sum + m
        feats.append(m - self.mean_sum / (k + self.n))
        return np.concatenate(feats)

    def _cov(self, x):
        x = x - x.mean(-1, keepdims=True)
        C = (x.astype(np.float64) @ x.T.astype(np.float64)) / x.shape[-1]
        s = float(self.p["shrink"])
        return (1 - s) * C + s * np.trace(C) / len(C) * np.eye(len(C))

    def _features(self, x):
        if self.car:
            x = x - x.mean(0, keepdims=True)
        if self.nsub:
            return self._features_sub(x)
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
        self.z = f  # scaled features of the last window (TSEM view)
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

    def __init__(self, p, dev="cpu"):
        t = {k: torch.from_numpy(np.ascontiguousarray(v)).to(dev) for k, v in p.items()
             if k not in ("align", "k", "G", "scale", "shrink")}
        self.t, self.dev = t, dev
        # CombinedConv: temporal and spatial filters merged into one kernel
        self.weight = ((t["time_w"] * t["spat_w"].permute(1, 0, 2, 3))
                       .sum(0).unsqueeze(1))
        self.bias = t["spat_w"].squeeze().sum(-1).mm(
            t["time_b"].unsqueeze(-1)).squeeze()

    @torch.no_grad()
    def proba(self, X):
        t = self.t
        h = X.unsqueeze(1).transpose(2, 3)
        h = F.conv2d(h, self.weight, self.bias)            # (B, 40, T', 1)
        h = F.batch_norm(h, t["bn_mean"], t["bn_var"], t["bn_w"], t["bn_b"],
                         False, 0.0, 1e-5)
        h = F.avg_pool2d(h * h, (75, 1), (15, 1))
        h = torch.log(torch.clamp(h, min=1e-6))
        h = F.conv2d(h, t["cls_w"], t["cls_b"])[:, :, 0, 0]
        return torch.softmax(h.double(), 1)


class Jit:
    """Stateless member shipped as a TorchScript file in the submission
    folder (key ``jit_file``): aligned (B, C, T) float32 -> logits (B, 3).
    New architectures can then change weight files only."""

    def __init__(self, p, dev="cpu", root="."):
        self.m = torch.jit.load(str(Path(root) / str(p["jit_file"])), map_location=dev).eval()
        self.dev = dev

    def __deepcopy__(self, memo):  # stateless: session copies share the module
        return self

    @torch.no_grad()
    def proba(self, X):
        h = self.m(X)
        return torch.softmax(h.double(), 1)


class SJit(Jit):
    """Session-stateful TorchScript member (key ``sjit_file``): its batch
    norms normalise window t with the causal statistics of the session's
    windows 0..t, carried in buffers named s1/s2/n (experiments/session_bn.py
    StreamBN). Each session copy owns its module; reset_state zeroes them."""

    def __init__(self, p, dev="cpu", root="."):
        super().__init__({"jit_file": p["sjit_file"]}, dev, root)

    def __deepcopy__(self, memo):
        new = copy.copy(self)
        new.m = copy.deepcopy(self.m, memo)
        return new

    def reset_state(self):
        for name, b in self.m.named_buffers():
            if name.rsplit(".", 1)[-1] in ("s1", "s2", "n"):
                b.zero_()


_FBC_SJITS = frozenset(
    f"{family}_s{seed}.pt"
    for family in ("sbn_k2", "sbn_k5", "sbn_fbcnet_raw_k20")
    for seed in range(6)
)
_FBL_SJITS = frozenset(f"sbn_fblight_ea_rot05_k5_s{seed}.pt" for seed in range(6))
_SBN_FIR_LENGTHS = (199, 199, 133, 99, 81, 67, 57, 51, 45)
_SBN_FIR_SHA256 = "607a44ee25576803092c03d77feb67411fe7a2266475fc730ae7ae28c81506c6"


class SharedFIRSJit(SJit):
    """Incumbent FBC/FBLight traces with one identical FIR bank per input group."""

    def __init__(self, p, dev="cpu", root="."):
        super().__init__(p, dev, root)
        name = str(p["sjit_file"])
        self.fblight = name in _FBL_SJITS
        expected = "FBLightConvNet" if self.fblight else "FBCNet"
        if self.m.original_name != expected:
            raise ValueError(f"{name}: expected a {expected} TorchScript module")
        bank = self.m.spectral_filtering
        if bank.original_name != "FilterBankLayer":
            raise ValueError(f"{name}: expected the TorchScript FilterBankLayer")
        filters = tuple(bank.fir_list.named_parameters())
        contract = tuple((key, tuple(f.shape), f.dtype) for key, f in filters)
        expected_contract = tuple((str(i), (n,), torch.float32) for i, n in enumerate(_SBN_FIR_LENGTHS))
        if contract != expected_contract:
            raise ValueError(f"{name}: unexpected FIR filter shapes, order, or dtype")
        data = b"".join(f.detach().cpu().numpy().astype("<f4", copy=False).tobytes() for _, f in filters)
        if hashlib.sha256(data).hexdigest() != _SBN_FIR_SHA256:
            raise ValueError(f"{name}: FIR coefficients do not match the incumbent bank")

    def bank(self, X):
        return self.m.spectral_filtering(X)

    @torch.no_grad()
    def proba_from_bank(self, Y):
        # Mirror the two archived forwards after spectral_filtering.
        h = self.m.spatial_conv(Y)
        if self.fblight:
            h = h[..., :250].reshape(Y.shape[0], 32, -1, 250)
            h = self.m.attn_conv(self.m.temporal_layer(h))
        else:
            h = self.m.temporal_layer(h.view(h.shape[0], h.shape[1], 4, 120))
        logits = self.m.final_layer(self.m.flatten_layer(h))
        return torch.softmax(logits.double(), 1)


def _sjit(p, dev, root):
    """SharedFIRSJit for the fingerprinted incumbent traces, otherwise the generic SJit (same outputs, slower)."""
    if str(p["sjit_file"]) in _FBC_SJITS or str(p["sjit_file"]) in _FBL_SJITS:
        try:
            return SharedFIRSJit(p, dev, root)
        except ValueError:
            pass
    return SJit(p, dev, root)


_POOL, _POOL_PID = None, None


def _submit(fn, *args):
    """fn(*args) on one process-local worker thread (a forked process cannot reuse its parent's thread)."""
    global _POOL, _POOL_PID
    if _POOL is None or _POOL_PID != os.getpid():
        _POOL, _POOL_PID = ThreadPoolExecutor(max_workers=1, thread_name_prefix="eeg-tangent"), os.getpid()
    return _POOL.submit(fn, *args)


def _fft_bank(x, fir, xf=None, ff=None):
    """"Same"-padded FIR filtering by FFT for every filter of a bank,
    (B, C, T) -> (B, K, C, T). ``xf`` caches input spectra per FFT length
    (shareable by all banks filtering the same input), ``ff`` filter
    spectra per (filter, length) (computed once per net)."""
    n, out = x.shape[-1], []
    xf = {} if xf is None else xf
    ff = {} if ff is None else ff
    for i, f in enumerate(fir):
        m = n + f.shape[-1] - 1
        if m not in xf:
            xf[m] = torch.fft.rfft(x, m)
        if (i, m) not in ff:
            ff[i, m] = torch.fft.rfft(f, m)
        s = (m - n) // 2
        out.append(torch.fft.irfft(xf[m] * ff[i, m], m)[..., s:s + n])
    return torch.stack(out, 1)


class FBC:
    """Stateless FBCNet (braindecode layout, plain torch, float64): FIR
    filter bank, grouped spatial filters, batch norm, SiLU, log-variance over
    ``stride`` segments, linear head. Weights are the effective (max-norm
    constrained) values; ``proba`` maps (B, C, T) to (B, 3)."""

    def __init__(self, p, dev="cpu"):
        self.t = {k: torch.from_numpy(np.ascontiguousarray(v)).double().to(dev)
                  for k, v in p.items() if k.startswith("fbc_")}
        n = int(p["fbc_n_bands"])
        self.fir = [self.t[f"fbc_fir{i}"] for i in range(n)]
        self.bank_key = b"".join(np.asarray(p[f"fbc_fir{i}"], np.float64).tobytes()
                                 for i in range(n))
        self.dev = dev
        self.ff = {}  # filter spectra, filled on first use
        self.stride = int(p["fbc_stride"]) if "fbc_stride" in p else None

    def bank(self, X, xf=None):
        """``xf``: input spectra shared by the nets of one input stream."""
        return _fft_bank(X, self.fir, xf, self.ff)

    @torch.no_grad()
    def proba(self, X, Y=None):
        """``Y``: precomputed ``bank(X)``, shared by nets with the same FIR."""
        t = self.t
        Y = self.bank(X) if Y is None else Y
        K, C = Y.shape[1], Y.shape[2]
        Z = torch.einsum("kfc,bkct->bkft", t["fbc_W"].view(K, -1, C), Y)
        Z = Z.reshape(X.shape[0], -1, X.shape[-1]) + t["fbc_bw"][None, :, None]
        Z = (Z - t["fbc_rm"][None, :, None]) / torch.sqrt(
            t["fbc_rv"][None, :, None] + t["fbc_eps"])
        Z = F.silu(Z * t["fbc_g"][None, :, None] + t["fbc_be"][None, :, None])
        s = self.stride
        V = Z.view(Z.shape[0], Z.shape[1], s, Z.shape[-1] // s).var(dim=3)
        V = torch.log(V.clamp(1e-6, 1e6)).flatten(1)
        return torch.softmax(V @ t["fbc_L"].T + t["fbc_lb"], 1)


class FBL(FBC):
    """Stateless FBLightConvNet (braindecode layout): FBCNet's FIR bank, a
    full spatial conv over bands, batch norm, ELU, log-variance of the first
    ``fbl_win`` samples, linear head (its one-tap attention is the identity)."""

    def __init__(self, p, dev="cpu"):
        super().__init__(p, dev)
        self.win = int(p["fbl_win"])
        self.eps = float(p["fbc_eps"])

    @torch.no_grad()
    def proba(self, X, Y=None):
        t = self.t
        Y = self.bank(X) if Y is None else Y
        Z = torch.einsum("fkc,bkct->bft", t["fbc_W"], Y) + t["fbc_bw"][None, :, None]
        Z = F.batch_norm(Z, t["fbc_rm"], t["fbc_rv"], t["fbc_g"], t["fbc_be"],
                         False, 0.0, self.eps)
        V = torch.log(F.elu(Z)[..., :self.win].var(dim=2).clamp(1e-6, 1e6))
        return torch.softmax(V @ t["fbc_L"].T + t["fbc_lb"], 1)


class Sinc:
    """Stateless SincShallowNet (braindecode layout, plain torch, float64):
    the trained sinc FIRs ("same" padding), batch norm, depthwise spatial
    conv, batch norm, ELU, average pooling, linear head."""

    def __init__(self, p, dev="cpu"):
        self.t = {k: torch.from_numpy(np.ascontiguousarray(v)).double().to(dev)
                  for k, v in p.items() if k.startswith("sinc_")}
        self.pool = tuple(int(v) for v in p["sinc_pool"])
        self.eps = float(p["sinc_eps"])
        self.dev = dev

    @torch.no_grad()
    def proba(self, X):
        t = self.t
        bn = lambda h, i: F.batch_norm(h, t[f"sinc_rm{i}"], t[f"sinc_rv{i}"], t[f"sinc_g{i}"],
                                       t[f"sinc_b{i}"], False, 0.0, self.eps)
        h = X.unsqueeze(1)
        h = bn(F.conv2d(h, t["sinc_f"], padding="same"), 1)        # (B, F, C, T)
        h = F.conv2d(h, t["sinc_dw"], groups=h.shape[1])          # (B, 2F, 1, T)
        h = F.avg_pool2d(F.elu(bn(h, 2)), (1, self.pool[0]), (1, self.pool[1]))
        z = h.flatten(1) @ t["sinc_L"].T + t["sinc_lb"]
        return torch.softmax(z, 1)


def _early(T, t):  # early-boost ramp: 1 up to t = T, linear to 0 at t = 2 T
    return min(max((2 * T - t) / T, 0.0), 1.0)


class SessionEM:
    """Causal per-session class templates in the frontal slow-potential view
    (experiments/session_em.py): window t uses windows < t only."""

    def __init__(self, p):
        self.chans = [p[k].astype(int) for k in ("em_hl", "em_hr", "em_vt", "em_vb")]
        self.lam, self.n0, self.m0 = float(p["em_lam"]), float(p["em_n0"]), float(p["em_m0"])
        self.shrink, self.iters, self.nmin = float(p["em_shrink"]), int(p["em_iters"]), int(p["em_nmin"])
        self.bins = int(p["em_bins"])
        self.ev_v = [p[k].astype(int) for k in ("em_ev_vt", "em_ev_vb")]  # R19e: causal ocular-event rates
        self.ev_g, self.ev_k = p["em_ev_g"].astype(np.float64), float(p["em_ev_k"])
        self.ev_q, self.ev_bins = p["em_ev_q"].astype(np.float64), int(p["em_ev_bins"])
        self.ev_sos = butter(4, 8, btype="low", fs=SFREQ, output="sos")
        self.boost = float(p.get("em_boost", 0.0))  # gated early boost (GEB, 7 Nov)
        if self.boost:
            self.boost_T, self.g_lo, self.g_hi = (float(p[k]) for k in ("eb_T", "eb_gate_lo", "eb_gate_hi"))
        self.reset_state()

    def reset_state(self):
        self.A, self.B, self.ev_hist = [], [], []
        self.msum, self.gates = 0.0, []

    def _gate(self, t):
        """GEB gate of window t: 1 for a weak session .. 0 for a strong one, from the mean input
        margin (top1 - top2) of windows < t; 1 at t = 0 and without the boost."""
        if not self.boost or t == 0:
            return 1.0
        return min(max((self.g_hi - self.msum / t) / (self.g_hi - self.g_lo), 0.0), 1.0)

    def events(self, x):
        n = (x.shape[-1] - 1) // self.ev_bins * self.ev_bins
        h = x[self.chans[0]].mean(0) - x[self.chans[1]].mean(0)
        v = x[self.ev_v[0]].mean(0) - x[self.ev_v[1]].mean(0)
        a = np.abs(np.stack([np.diff(sosfiltfilt(self.ev_sos, s))[:n] for s in (h, v)]))
        self.ev_hist.append(np.median(a, axis=1))
        m = (self.ev_k * self.ev_g + np.sum(self.ev_hist, 0)) / (self.ev_k + len(self.ev_hist))
        f = [(a[i] > q * m[i]).reshape(self.ev_bins, -1).mean(-1) for i in range(2) for q in self.ev_q]
        return np.log(np.concatenate(f) + 1e-3)

    def features(self, x):
        x = x.astype(np.float64)
        t = x.shape[-1] // self.bins * self.bins
        hl, hr, vt, vb = (x[c].mean(0)[:t].reshape(self.bins, -1).mean(-1) for c in self.chans)
        f = np.concatenate([hl - hr, vt - vb])
        return np.concatenate([f, self.events(x)])

    @staticmethod
    def _lik(Z, mu, Ci):
        W = Ci @ mu.T
        s = Z @ W - 0.5 * (mu * W.T).sum(1)
        s = s - s.max(-1, keepdims=True)
        g = np.exp(s)
        return g / g.sum(-1, keepdims=True)

    def step(self, x, p):
        f, t, out = self.features(x), len(self.A), p
        self.gates.append(self._gate(t))
        if t >= self.nmin:
            try:
                out = self._adapt(np.asarray(self.A), np.asarray(self.B), f, p, t)
            except np.linalg.LinAlgError:  # degenerate view (e.g. flat frontal channels)
                pass
            if not np.isfinite(out).all():
                out = p
        self.A.append(f)
        self.B.append(p)
        s = np.sort(p)
        self.msum += s[-1] - s[-2]
        return out

    def _adapt(self, A, B, f, p, t):
        # sd floor: a column constant over the past (an event rate that never
        # fired) must not turn float rounding into huge z scores
        m, sd = A.mean(0), np.maximum(A.std(0), 1e-3)
        Z, z = (A - m) / sd, (f - m) / sd
        Q = B
        for _ in range(self.iters):
            mu = (Q / (Q.sum(0, keepdims=True) + self.m0)).T @ Z
            C = np.cov((Z - Q @ mu).T)
            C = (1 - self.shrink) * C + self.shrink * np.eye(len(C)) * np.trace(C) / len(C)
            Ci = np.linalg.inv(C)
            Q = B * self._lik(Z, mu, Ci) ** self.lam
            Q = Q / Q.sum(1, keepdims=True)
        e = self.lam * t / (t + self.n0)
        if self.boost:
            e *= 1 + self.boost * (_early(self.boost_T, t) * self.gates[t])
        q = p * self._lik(z[None], mu, Ci)[0] ** e
        return q / q.sum()


class TSEM:
    """Causal per-session class templates in a fixed PCA view of the first
    tangent member's scaled features (u = (z - tsem_m) @ tsem_V): window t
    uses windows < t, seeded per class by the most confident input posteriors
    (the SessionEM output, not its own output); LDA likelihood ** lam t / (t + n0)."""

    def __init__(self, p):
        self.lam, self.rho, self.shrink = (float(p[f"tsem_{k}"]) for k in ("lam", "rho", "shrink"))
        self.m0, self.n0, self.nmin = float(p["tsem_m0"]), float(p["tsem_n0"]), int(p["tsem_nmin"])
        self.m, self.V = p["tsem_m"].astype(np.float64), p["tsem_V"].astype(np.float64)
        self.boost = float(p.get("tsem_boost", 0.0))  # GEB, with the gate of this tail's SessionEM
        if self.boost:
            self.boost_T = float(p["eb_T"])
        self.gate_src = None  # that SessionEM (wired by Ensemble; Route3.bind copies the pair)
        self.reset_state()

    def reset_state(self):
        self.U, self.conf, self.lab = [], [], []

    def view(self, z):
        return (z - self.m) @ self.V

    def step(self, u, q):
        t, out = len(self.U), q
        if self.lam > 0 and t >= self.nmin:
            try:
                out = self._adapt(np.asarray(self.U), np.asarray(self.conf), np.asarray(self.lab), u, q, t)
            except np.linalg.LinAlgError:
                out = q
            if not np.isfinite(out).all():
                out = q
        self.U.append(u)
        self.conf.append(q.max())
        self.lab.append(q.argmax())
        return out

    def _adapt(self, A, conf, lab, u, q, t):
        m, sd = A.mean(0), np.maximum(A.std(0), 1e-3)
        Z, z = (A - m) / sd, (u - m) / sd
        k = max(2, int(self.rho * t / 3))
        sel = [i[np.argsort(-conf[i])[:k]] for i in (np.flatnonzero(lab == c) for c in range(3))]
        if min(len(i) for i in sel) < 2:
            return q
        mu = np.stack([Z[i].sum(0) / (len(i) + self.m0) for i in sel])
        C = np.cov(np.concatenate([Z[i] - mu[c] for c, i in enumerate(sel)]).T)
        C = (1 - self.shrink) * C + self.shrink * np.eye(len(C)) * np.trace(C) / len(C)
        e = self.lam * t / (t + self.n0)
        if self.boost:
            e *= 1 + self.boost * (_early(self.boost_T, t) * self.gate_src.gates[t])
        qq = q * SessionEM._lik(z[None], mu, np.linalg.inv(C))[0] ** e
        return qq / qq.sum()


class Rebalance:
    """Causal label-free prior correction: window t's mix divided by m ** alpha, m being the mean
    mix of the session's last W windows blended with a uniform prior weighted as k windows."""

    def __init__(self, alpha, k, W):
        assert alpha and W > 0  # the uploaded packages: alpha 1 / 1.15, k 5, W 30
        self.alpha, self.k, self.W = alpha, k, W
        self.reset_state()

    def reset_state(self):
        self.hist = []

    def step(self, p):
        h = self.hist[-self.W:]
        m = (self.k / len(p) + np.asarray(h).sum(0)) / (self.k + len(h))
        q = p / m ** self.alpha
        self.hist = (self.hist + [p])[-self.W:]
        return q / q.sum()


def _softmax_rows(D):  # sklearn.utils.extmath.softmax (LogisticRegression.predict_proba)
    D = D - np.max(D, axis=1).reshape((-1, 1))
    np.exp(D, D)
    return D / np.sum(D, axis=1).reshape((-1, 1))


def _rz(A, x, clip, floor):
    """x standardised by the per-feature median / (1.4826 MAD) of the rows of A, clipped at +-clip."""
    med = np.median(A, axis=0)
    s = 1.4826 * np.median(np.abs(A - med), axis=0)
    return np.clip((x - med) / np.maximum(s, floor), -clip, clip)


class Bio:
    """step(x) -> (person index, ramp r in [0, 1]); r = clip((score - tau) / ramp, 0, 1) from window t0 on."""

    def __init__(self, p):
        f = lambda k: np.asarray(p[k], dtype=np.float64)
        self.mu, self.sd, self.V, self.B, self.W, self.S = (f(k) for k in ("bio_mu", "bio_sd", "bio_V", "bio_B", "bio_W", "bio_S"))
        self.G, self.m = f("bio_gal"), np.asarray(p["bio_m"]).astype(int)
        self.tau, self.ramp, self.t0 = float(p["bio_tau"]), float(p["bio_ramp"]), int(p["bio_t0"])
        self.win, self.d = np.hanning(120), self.V.shape[1]
        self.A = {m: self.B + self.W / m for m in np.unique(self.m)}
        self.gAg = np.array([g @ np.linalg.solve(self.A[m], g) for g, m in zip(self.G, self.m)])
        self.ldA = {m: np.linalg.slogdet(A)[1] for m, A in self.A.items()}
        self.reset_state()

    def reset_state(self):
        self.sum, self.n, self.last = 0.0, 0, None

    def feat(self, x):
        x = np.asarray(x, dtype=np.float64)
        x = x - x.mean(0, keepdims=True)
        x = x - x.mean(1, keepdims=True)
        seg = np.stack([x[:, s:s + 120] for s in range(0, 361, 60)], 1) * self.win
        L = np.log((np.abs(np.fft.rfft(seg, axis=-1)) ** 2).mean(1)[:, 2:41] + 1e-12)
        return (L - L.mean(1, keepdims=True)).ravel()

    def scores(self):
        n, d = self.n, self.d
        x = ((self.sum / n - self.mu) / self.sd) @ self.V
        C = self.B + self.W + self.S / n
        xCx, ldC = x @ np.linalg.solve(C, x), np.linalg.slogdet(C)[1]
        out = np.empty(len(self.G))
        for m, A in self.A.items():
            same = np.block([[A, self.B], [self.B, C]])
            ld = np.linalg.slogdet(same)[1]
            for i in np.flatnonzero(self.m == m):
                v = np.concatenate([self.G[i], x])
                out[i] = (-0.5 * (v @ np.linalg.solve(same, v) + ld) + 0.5 * (self.gAg[i] + xCx + self.ldA[m] + ldC)) / d
        return out

    def step(self, x):
        self.sum = self.sum + self.feat(x)
        self.n += 1
        self.last = None
        if self.n - 1 < self.t0:
            return -1, 0.0
        s = self.scores()
        if not np.isfinite(s).all():
            return -1, 0.0
        j = int(np.argmax(s))
        r = float(np.clip((s[j] - self.tau) / self.ramp, 0, 1))
        self.last = (j, float(s[j]), r)
        return j, r


class OwnStage:
    """v5 (12 Nov): personal members in the returning person's own space, as a final stage on routed windows.
    Per public person, fitted on their public sessions: T = tangent space of ``own_bands`` after recentring to the
    session's running mean covariance (the person's public mean as ``own_kref`` pseudo-windows), standardised
    multinomial LR; C = FBCSP log-variances of the same recentred covariances, LDA; S = slow time courses
    (0-4 Hz, 8 per second, standardised over the session's windows so far), LDA.
    observe(x) runs on every window (session sums); log_members(j) is the mean clipped log posterior of T, C, S
    for person j; the routed output becomes out * exp(beta * r * log_members)."""

    def __init__(self, p):
        f = lambda k: np.asarray(p[k], dtype=np.float64)
        self.beta, self.kref = float(p["own_beta"]), float(p["own_kref"])
        self.sos = [butter(4, [float(lo), float(hi)], btype="band", fs=120, output="sos") for lo, hi in f("own_bands")]
        self.sos_slow = butter(2, 4, btype="low", fs=120, output="sos")
        self.prior, self.csp = f("own_prior"), f("own_csp")  # (P, B, C, C), (P, B, C, F)
        self.t_mu, self.t_sd, self.t_W, self.t_c = f("own_t_mu"), f("own_t_sd"), f("own_t_W"), f("own_t_c")
        self.c_W, self.c_c, self.s_W, self.s_c = f("own_c_W"), f("own_c_c"), f("own_s_W"), f("own_s_c")
        n = self.prior.shape[-1]
        self.iu = np.triu_indices(n)
        self.wt = np.where(self.iu[0] == self.iu[1], 1.0, np.sqrt(2.0))
        self.eye = np.eye(n) / n
        self.reset_state()

    def reset_state(self):
        self.n, self.csum, self.s1, self.s2, self.cur = 0, 0.0, 0.0, 0.0, None

    def observe(self, x):
        x = np.asarray(x, dtype=np.float64)
        self.cur = None  # a degenerate window (non-finite, or a flat channel) is not added to the sums and not staged
        sd = x.std(1)
        if not (np.isfinite(x).all() and sd.min() > 1e-9 * max(float(sd.max()), 1e-30)):
            return
        xc = x - x.mean(0, keepdims=True)
        C = []
        for sos in self.sos:
            b = sosfiltfilt(sos, xc, axis=-1)
            c = b @ b.T / b.shape[-1]
            C.append(c + 1e-3 * np.trace(c) * self.eye)
        C = np.stack(C)
        sl = sosfiltfilt(self.sos_slow, x, axis=-1)[:, 7::15].ravel()
        self.csum, self.s1, self.s2 = self.csum + C, self.s1 + sl, self.s2 + sl ** 2
        self.n += 1
        self.cur = (C, sl)

    def log_members(self, j):
        C, sl = self.cur
        n = self.n
        w, V = np.linalg.eigh((self.kref * self.prior[j] + self.csum) / (self.kref + n))
        W = np.einsum("bij,bj,bkj->bik", V, 1 / np.sqrt(w), V)
        Cw = W @ C @ W
        w, V = np.linalg.eigh(Cw)
        Lg = np.einsum("bij,bj,bkj->bik", V, np.log(w), V)
        t = (Lg[:, self.iu[0], self.iu[1]] * self.wt).ravel()
        pt = _softmax_rows(((t - self.t_mu[j]) / self.t_sd[j])[None] @ self.t_W[j].T + self.t_c[j])[0]
        fc = np.log(np.einsum("bcf,bcd,bdf->bf", self.csp[j], Cw, self.csp[j])).ravel()
        pc = _softmax_rows(fc[None] @ self.c_W[j].T + self.c_c[j])[0]
        m = self.s1 / n
        z = (sl - m) / np.sqrt(np.maximum(self.s2 / n - m ** 2, 1e-9)) if n >= 4 else np.zeros_like(sl)
        ps = _softmax_rows(z[None] @ self.s_W[j].T + self.s_c[j])[0]
        return sum(np.log(np.clip(q, 1e-3, 1)) for q in (pt, pc, ps)) / 3


class Route3:
    """Returning-person routing v3 (7 Nov). Gate on the SHADOW tail = this
    ensemble's own unrouted rebalancing -> SessionEM -> TSEM output: view u = [TSEM view,
    (z - tsem_m) @ route_V_hi] (256 PCs); from window t_start the class contrast of
    windows < t (u robust-standardised: median / 1.4826 MAD, clipped at +-route_clip; the
    top max(2, rho t / 3) shadow outputs per argmax class), projected off span{route_pop,
    route_b} (route_Q), is compared by cosine with each public person's projected gallery;
    r = clip((score - tau) / ramp, 0, 1). Routed tail (own rebalancing state and copies of
    the SessionEM / TSEM stages): the mix with core1 (share a) partly replaced by the best
    person's PS1 head h (LR on u standardised over windows <= t) at g times its weight,
    mix' = (mix - r a core1 + g r a h) / (1 - r a + g r a); output = routed tail (equal to
    the shadow tail until the first routed window). The gate never reads the routed tail."""

    def __init__(self, p):
        f = lambda k: np.asarray(p[k], dtype=np.float64)
        assert int(p["route_v3"]) == 1
        self.G, self.Qb, self.W, self.c = f("route_gal"), f("route_Q"), f("route_ps1_W"), f("route_ps1_c")
        self.tau, self.ramp, self.rho, self.clip = (float(p[f"route_{k}"]) for k in ("tau", "ramp", "rho", "clip"))
        self.g, self.t_start = float(p["route_slot_g"]), int(p["route_t_start"])
        A = np.stack([self._perp(g) for g in self.G])
        self.Gn = A / np.linalg.norm(A, axis=1, keepdims=True)
        self.m, self.V = p["tsem_m"].astype(np.float64), p["route_V_hi"].astype(np.float64)
        self.bio = Bio(p) if "bio_V" in p else None  # v4: label-free identity gate (11 Nov)
        assert self.bio is None or len(self.bio.G) == len(self.W)  # one gallery row per personal head
        self.own = OwnStage(p) if "own_beta" in p else None  # v5: own-space personal stage (12 Nov)
        assert self.own is None or len(self.own.prior) == len(self.W)

    def bind(self, rb, em, tsem, a):
        """Routed tail: own copies of the rebalancing, SessionEM and TSEM stages."""
        self.rb, self.em, self.tsem = copy.deepcopy((rb, em, tsem))  # one copy: tsem.gate_src -> its own em
        self.a = float(a)
        self.reset_state()

    def report(self):
        """v5.1b (12 Nov): one stderr line per session on what the own-space stage did, so a silent no-op on the sealed
        run is visible in the logs (windows, routed, degenerate, staged, errors and the first error)."""
        lg = getattr(self, "log", None)
        if lg and lg["n"]:
            print(f"[own-stage] windows {lg['n']} routed {lg['routed']} degenerate {lg['degen']} staged {lg['staged']} "
                  f"errors {lg['err']}{' first: ' + lg['first'] if lg['first'] else ''}", file=sys.stderr, flush=True)
        self.log = dict(n=0, routed=0, degen=0, staged=0, err=0, first="")

    def __del__(self):
        try:
            self.report()
        except Exception:
            pass

    def reset_state(self):
        self.report()
        self.U, self.Qs, self.last = [], [], None
        for stage in (self.rb, self.em, self.tsem):
            stage.reset_state()
        if self.bio is not None:
            self.bio.reset_state()
        if self.own is not None:
            self.own.reset_state()
        self.stage = None

    def view(self, z):
        return (z - self.m) @ self.V

    def _perp(self, v):
        v = v.ravel()
        return v - self.Qb @ (self.Qb.T @ v)

    def scores(self, t):
        X, P = np.asarray(self.U[:t]), np.asarray(self.Qs[:t])
        lab, conf = P.argmax(1), P.max(1)
        Wt, q = np.zeros((t, 3)), max(2, int(self.rho * t / 3))
        for k in range(3):
            ik = np.flatnonzero(lab == k)
            Wt[ik[np.argsort(-conf[ik])[:q]], k] = 1
        n = Wt.sum(0)
        if (n < 1e-9).any():
            return None
        Z = _rz(X, X, self.clip, 1e-6)
        mu = (Wt.T @ Z) / n[:, None]
        a = self._perp(mu - mu.mean(0))
        na = np.linalg.norm(a)
        return self.Gn @ a / na if na > 1e-9 else np.zeros(len(self.Gn))

    def step(self, x, u, mix, core1, qs):
        """x: raw window (SessionEM input), u: 256-PC view, mix / core1: B2 mixture and core1
        probabilities before rebalancing, qs: shadow-tail output of this window."""
        u, mix, core1, qs = (np.asarray(v, dtype=np.float64) for v in (u, mix, core1, qs))
        t, p = len(self.U), mix
        self.last = None
        j, r = -1, 0.0
        if t >= self.t_start:
            try:
                s = self.scores(t)
                if s is not None and np.isfinite(s).all():
                    j = int(np.argmax(s))
                    r = float(np.clip((s[j] - self.tau) / self.ramp, 0, 1))
                    self.last = (j, s[j], r)
            except (np.linalg.LinAlgError, FloatingPointError, ValueError):
                j, r = -1, 0.0
        if self.bio is not None:  # v4: the identity gate sees every window; the larger ramp wins
            try:
                jb, rb = self.bio.step(x)
            except (np.linalg.LinAlgError, FloatingPointError, ValueError):
                jb, rb = -1, 0.0
            if rb > r:
                j, r = jb, rb
        if r > 0:
            try:
                A = np.asarray(self.U + [u])
                m = A.mean(0)
                z = (u - m) / np.maximum(A.std(0), 1e-3) if t >= 4 else u - m
                h = _softmax_rows(z[None] @ self.W[j].T + self.c[j])[0]
                o = (mix - r * self.a * core1 + self.g * r * self.a * h) / (1 - r * self.a + self.g * r * self.a)
                if np.isfinite(o).all():
                    p = o
            except (np.linalg.LinAlgError, FloatingPointError, ValueError):
                p = mix
        out = self.tsem.step(u[:self.tsem.V.shape[1]], self.em.step(x, self.rb.step(p)))
        if self.own is not None:  # v5: every window feeds the session sums; routed windows get the stage
            self.stage = None
            lg = self.log; lg["n"] += 1; lg["routed"] += int(r > 0)
            try:  # the stage is optional: any failure leaves the v4.1 output of this window
                with np.errstate(all="ignore"):
                    self.own.observe(x)
                    lg["degen"] += int(self.own.cur is None)
                    if r > 0 and self.own.cur is not None:
                        lm = self.own.log_members(j)
                        o = np.log(np.clip(out, 1e-9, 1)) + self.own.beta * r * lm
                        o = np.exp(o - o.max())
                        o = o / o.sum()
                        if np.isfinite(lm).all() and np.isfinite(o).all():
                            out, self.stage = o, (j, r, lm)
                            lg["staged"] += 1
            except Exception as e:  # v5.1b: counted and reported at the session's end
                lg["err"] += 1; lg["first"] = lg["first"] or repr(e)[:200]
        self.U.append(u)
        self.Qs.append(qs)
        return out


def _input_key(p):
    keys = ("align", "k", "G", "scale", "shrink")
    return b"".join(np.ascontiguousarray(p[k]).tobytes() for k in keys if k in p)


class Ensemble:
    """Weighted mean of member probabilities.

    Network inputs are computed window by window (causal session state);
    nets are stateless, so they then run once over the whole batch. Nets
    with identical alignment parameters share one input stream.
    """

    def __init__(self, tangent, nets, weights, rb=None, dev="cpu", em=None,
                 root=".", tangent2=None, tsem=None, route=None):
        self.tangent, self.w, self.rb, self.em, self.tsem = tangent, weights / weights.sum(), rb, em, tsem
        if em is not None and tsem is not None:
            tsem.gate_src = em  # GEB: TSEM's early boost uses SessionEM's per-window gate
        self.route = route  # gate on TSEM's output and views, routed tail of its own
        if route is not None:
            assert None not in (rb, em, tsem) and tsem.V.shape[1] + route.V.shape[1] == route.G.shape[2]
            route.bind(rb, em, tsem, self.w[0])
        self.tangent2 = tangent2  # optional second tangent member, weight w[1 + len(nets)]
        self.w2 = self.w[1 + len(nets)] if tangent2 is not None else 0.0
        self.perm = None  # set by Solver.load_model from meta["ch_names"]
        self.dev = dev
        self.groups = {}
        for i, p in enumerate(nets):
            key = _input_key(p)
            if key not in self.groups:
                self.groups[key] = (InputStream(p), [])
            if self.w[1 + i] == 0:
                continue
            net = (_sjit(p, dev, root) if "sjit_file" in p else
                   Jit(p, dev, root) if "jit_file" in p else
                   Sinc(p, dev) if "sinc_f" in p else
                   FBL(p, dev) if "fbl_win" in p else
                   FBC(p, dev) if "fbc_n_bands" in p else Shallow(p, dev))
            self.groups[key][1].append((i, net))
        self.groups = {key: group for key, group in self.groups.items() if group[1]}

    def reset_state(self):
        for stage in (self.tangent, self.tangent2, self.rb, self.em, self.tsem, self.route):
            if stage is not None:
                stage.reset_state()
        for stream, nets in self.groups.values():
            stream.reset_state()
            for _, net in nets:  # session-stateful members only (SJit)
                if hasattr(net, "reset_state"):
                    net.reset_state()

    @torch.no_grad()
    def _tangent_batch(self, X):
        """Tangent members (CPU), window by window: probabilities and the TSEM / routing views."""
        first, second, views, views_hi = [], [], [], []
        for x in X:  # causal state updates, in acquisition order
            first.append(self.tangent.proba(x))
            if self.tsem is not None:
                views.append(self.tsem.view(self.tangent.z))
            if self.route is not None:
                views_hi.append(self.route.view(self.tangent.z))
            if self.tangent2 is not None:
                second.append(self.tangent2.proba(x))
        return first, second, views, views_hi

    def _neural_batch(self, X):
        """Network members: causal input streams window by window, then each net once over the batch.
        Returns (member index, host posteriors) pairs in member order."""
        inputs = {key: [] for key in self.groups}
        for x in X:
            for key, (stream, _) in self.groups.items():
                inputs[key].append(stream(x))
        member_ids, member_probs = [], []
        for key, (_, nets) in self.groups.items():
            H = torch.from_numpy(np.stack(inputs[key]))
            banks, xf, device_inputs = {}, {}, {}
            shared_fir = None
            for i, net in nets:
                dtype = torch.float64 if isinstance(net, (FBC, Sinc)) else torch.float32
                if dtype not in device_inputs:
                    device_inputs[dtype] = H.to(dtype=dtype).to(self.dev)
                h = device_inputs[dtype]
                if isinstance(net, SharedFIRSJit):
                    if shared_fir is None:
                        shared_fir = net.bank(h)
                    prob = net.proba_from_bank(shared_fir)
                elif isinstance(net, FBC):
                    if net.bank_key not in banks:
                        banks[net.bank_key] = net.bank(h, xf)
                    prob = net.proba(h, banks[net.bank_key])
                else:
                    prob = net.proba(h)
                member_ids.append(i)
                member_probs.append(prob)
        if member_probs:
            return list(zip(member_ids, torch.stack(member_probs).cpu().numpy()))
        return []

    @torch.no_grad()
    def mixed_proba(self, X):
        """Weighted member probabilities, (B, 3); advances session state. The tangent members run on
        the worker thread while the nets run here (disjoint state); both finish before this returns."""
        X = X.detach().cpu().numpy() if torch.is_tensor(X) else np.asarray(X)
        X = np.nan_to_num(X.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        tangent = _submit(self._tangent_batch, X)
        try:
            members = self._neural_batch(X)
        finally:  # never let tangent state advance after returning, even on an error here
            wait((tangent,))
        first, second, self.views, self.views_hi = tangent.result()
        self.core1 = np.stack(first)  # core1's probabilities (routed tail, v3)
        P = self.w[0] * self.core1
        for i, prob in members:  # the lean code's CPU accumulation order and dtype
            P = P + self.w[1 + i] * prob
        if self.tangent2 is not None:
            P = P + self.w2 * np.stack(second)
        return P

    def predict(self, X):
        if self.perm is not None:  # sealed montage in another order: map by name
            X = X.detach().cpu().numpy() if torch.is_tensor(X) else np.asarray(X)
            Xn = np.zeros((len(X), len(self.perm), X.shape[-1]), np.float32)
            Xn[:, self.perm >= 0] = X[:, self.perm[self.perm >= 0]]
            X = Xn
        P = self.mixed_proba(X)
        P[~np.isfinite(P).all(1)] = 1.0 / P.shape[1]  # never let NaN enter session state
        Pmix = P.copy()  # routed tail input
        X = X.detach().cpu().numpy() if torch.is_tensor(X) else np.asarray(X)
        X = np.nan_to_num(X.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        # causal stages, each window by window in acquisition order
        if self.rb is not None:
            P = np.stack([self.rb.step(p) for p in P.copy()])
        if self.em is not None:
            P = np.stack([self.em.step(x, p) for x, p in zip(X, P)])
        if self.tsem is not None:
            P = np.stack([self.tsem.step(u, p) for u, p in zip(self.views, P)])
        if self.route is not None:  # P = shadow output; output = routed tail
            P = np.stack([self.route.step(x, np.r_[u, v], pm, c1, q) for x, u, v, pm, c1, q
                          in zip(X, self.views, self.views_hi, Pmix, self.core1, P)])
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
        # Nets run on the harness device (CUDA on the worker),
        # without TF32 so GPU and CPU agree to float32 precision
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        dev = str(meta.get("device", "cpu"))
        tss = []
        for pre in ("ts_", "ts2_"):
            ts = {k[len(pre):]: v for k, v in z.items() if k.startswith(pre)}
            if ts:
                ts["log_glob"] = list(ts["log_glob"])
                ts["bands"] = [tuple(b) for b in ts["bands"]]
                tss.append(AlignedTangent(ts))
        nets = []
        for i in range(int(z["n_nets"])):
            pre = f"net{i}_"
            nets.append({k[len(pre):]: v for k, v in z.items() if k.startswith(pre)})
        alpha = float(z.get("rb_alpha", 0.0))
        model = Ensemble(tss[0], nets, z["member_weights"].astype(np.float64),
                         rb=Rebalance(alpha, float(z["rb_k"]), int(z["rb_w"])) if alpha else None, dev=dev,
                         em=SessionEM(z) if "em_lam" in z else None,
                         root=meta["submission_dir"], tangent2=tss[1] if len(tss) > 1 else None,
                         tsem=TSEM(z) if "tsem_lam" in z else None,
                         route=Route3(z) if "route_tau" in z else None)
        model.perm = channel_map(meta.get("ch_names"))
        return model


def channel_map(names):
    """Index of each training channel in the runtime montage (-1: missing,
    zero-filled), or None when the runtime order is the training order or
    the names are unusable (then inputs stay positional)."""
    try:
        names = [str(c).strip().upper() for c in names]
    except TypeError:
        return None
    train = [c.upper() for c in CH_NAMES]
    if names == train or len(set(names) & set(train)) < len(train) // 2:
        return None
    return np.array([names.index(c) if c in names else -1 for c in train])
