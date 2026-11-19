"""Lane G shared pieces: public windows, pinned deliveries, the package module, causal feature streams.

The package module (submissions/lean_v66/submission.py inside the sealed cap250 ZIP) supplies TangentCore (core1 view) and
OwnStage, so the features and the stage that score the arms are the shipped code, not a copy. Nothing here reads a
task label for a held visit; labels enter only in fit() of the training visits.
"""
import importlib.util
import os
import sys
import zipfile
from pathlib import Path

import numpy as np

ROOT = Path(os.environ.get("NI26_ROOT", Path(__file__).resolve().parents[1]))
ORDERS = ("within0", "within1", "within2", "within3", "full0", "full1", "runs0", "runs1")
SPLITS = (("P04", (1,), 2), ("P04", (1,), 3), ("P04", (1, 2), 3), ("P10", (1,), 2), ("P10", (1,), 3), ("P10", (1, 2), 3))
NAMES = ["P%02d" % i for i in range(1, 11)]


def public():
    with np.load(ROOT / "data/proteus_windows.npz", allow_pickle=True) as z:
        return dict(X=z["X"], y=z["y"], person=np.array([s[-3:] for s in z["subject"]]), visit=z["session"].astype(int),
                    itf=z["interface"].astype(str))


def delivery(D, order, person, visit):
    """Row indices (into the public windows) of one held visit in the pinned delivery order (Lane A's rule)."""
    import re
    kind, k = re.fullmatch(r"(within|full|runs)(\d)", order).groups()
    with np.load(ROOT / "runs/shuffle_perms.npz", allow_pickle=False) as f:
        perm = f[f"{kind}|{k}|seen_session"]
    return perm[(D["person"][perm] == person) & (D["visit"][perm] == visit)]


def package(zip_path, work):
    """Extract weights.npz + submission.py of a package ZIP into ``work``; -> (module, weights dict)."""
    work = Path(work)
    work.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        for n in ("weights.npz", "submission.py"):
            if not (work / n).exists():
                (work / n).write_bytes(zf.read(n))
    import types  # the Solver base class is not used here: stub it, so no benchopt / official tree is needed
    base = types.ModuleType("benchmark_utils.base_solver")
    base.CompetSolver = object
    sys.modules.update({"benchmark_utils": types.ModuleType("benchmark_utils"), "benchmark_utils.base_solver": base})
    spec = importlib.util.spec_from_file_location("laneg_pkg", work / "submission.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    with np.load(work / "weights.npz", allow_pickle=False) as f:
        return mod, {k: f[k] for k in f.files}


def views(mod, z, X):
    """(n,256) core1 view of windows streamed causally in the given order, one fresh TangentCore (Ensemble._tangents)."""
    core = mod.TangentCore(mod.with_prefix(z, "ts_"))
    core.reset_state()
    m, V0, V1 = (np.asarray(z[k], np.float64) for k in ("tsem_m", "tsem_V", "route_V_hi"))
    out = []
    for x in np.nan_to_num(np.asarray(X, np.float32)):
        d = core.step(x)[1] - m
        out.append(np.r_[d @ V0, d @ V1])
    return np.asarray(out)


def strong(X, rng, shift=15, noise=0.2, drop=0.1):
    """The package's `strong` augmentation (finetune): one roll for the batch, noise at a share of each channel's sd,
    a share of channels zeroed; numpy generator."""
    Xs = np.roll(X, int(rng.integers(-shift, shift + 1)), -1)
    Xs = Xs + noise * Xs.std(-1, keepdims=True) * rng.standard_normal(Xs.shape)
    return (Xs * (rng.random((Xs.shape[0], Xs.shape[1], 1)) > drop)).astype(np.float32)


def soft(X, rng, shift=15, noise=0.2, gain=0.2):
    """Covariance-safe augmentation: roll, noise at a share of each channel's sd, per-channel gain lognormal(0, gain); no zeroing.
    gain 0 and noise > 0 = the noise-only form (g4)."""
    Xs = np.roll(X, int(rng.integers(-shift, shift + 1)), -1)
    Xs = Xs + noise * Xs.std(-1, keepdims=True) * rng.standard_normal(Xs.shape)
    return (Xs * np.exp(gain * rng.standard_normal((Xs.shape[0], Xs.shape[1], 1)))).astype(np.float32)


class OwnFeat:
    """Own-space window features exactly as OwnStage.observe: (bands, C, C) regularised band covariances and the slow
    time course (C * 32). Batch over windows (n, C, T)."""

    def __init__(self, z):
        from scipy.signal import butter
        self.sos = [butter(4, [float(lo), float(hi)], btype="band", fs=120.0, output="sos") for lo, hi in z["own_bands"]]
        self.slow = butter(2, 4, btype="low", fs=120.0, output="sos")
        self.n = z["own_prior"].shape[-1]

    def __call__(self, X):
        from scipy.signal import sosfiltfilt
        X = np.asarray(X, np.float64)
        xc = X - X.mean(1, keepdims=True)
        C = []
        for sos in self.sos:
            b = sosfiltfilt(sos, xc, axis=-1)
            c = np.einsum("nct,ndt->ncd", b, b) / b.shape[-1]
            C.append(c + 1e-3 * np.trace(c, axis1=1, axis2=2)[:, None, None] * np.eye(self.n)[None] / self.n)
        sl = sosfiltfilt(self.slow, X, axis=-1)[:, :, 7::15].reshape(len(X), -1)
        return np.stack(C, 1), sl  # (n, bands, C, C), (n, C*32)


def ba_cells(pred, y, itf):
    """Balanced accuracy (percent) per interface cell of one visit (complete classes), and their mean."""
    cells = {}
    for i in np.unique(itf):
        m = itf == i
        cells[str(i)] = 100.0 * np.mean([np.mean(pred[m & (y == k)] == k) for k in np.unique(y[m])])
    return cells, float(np.mean(list(cells.values())))
