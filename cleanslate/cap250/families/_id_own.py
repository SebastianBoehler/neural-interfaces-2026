"""Own-space personal members (studies/12nov/ownspace.py, ledger s37) fitted on explicit source rows only."""

import sys
from pathlib import Path

import numpy as np
from scipy.linalg import eigh
from scipy.signal import butter, sosfiltfilt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

BANDS = [(1, 4), (4, 8), (8, 13), (13, 20), (20, 30), (30, 40), (40, 58), (4, 40), (8, 30)]
IU = np.triu_indices(41)
KREF = 2.0
KEYS = ("prior", "csp", "t_mu", "t_sd", "t_W", "t_c", "c_W", "c_c", "s_W", "s_c")


def _check_reference():
    """The tangent and class members must stay the reference implementation; fail if its constants moved."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "studies/12nov"))
    text = (Path(__file__).resolve().parents[3] / "studies/12nov/ownspace.py").read_text()
    for token in ("BANDS = " + str(BANDS), "KREF = 2.0", "C=0.01, max_iter=500", "shrinkage=0.5", "shrinkage=0.7"):
        if token not in text:
            raise RuntimeError(f"ownspace.py differs from the cap250 identity recipe: {token}")


def _isqrt(m):
    w, v = np.linalg.eigh(m)
    return np.einsum("...ij,...j,...kj->...ik", v, 1 / np.sqrt(w), v)


def _logm(c):
    w, v = np.linalg.eigh(c)
    return np.einsum("...ij,...j,...kj->...ik", v, np.log(w), v)


class Space:
    """Band covariances and slow time courses of the source windows, as ownspace.py computes them."""

    def __init__(self, x_raw, y):
        _check_reference()
        self.y = y
        x = x_raw.astype(np.float64)
        xc = x - x.mean(1, keepdims=True)
        self.cov = []
        for lo, hi in BANDS:
            b = sosfiltfilt(butter(4, [lo, hi], btype="band", fs=120, output="sos"), xc, axis=-1)
            c = np.einsum("nct,ndt->ncd", b, b) / b.shape[-1]
            self.cov.append(c + 1e-3 * np.trace(c, axis1=1, axis2=2)[:, None, None] / 41 * np.eye(41))
        self.slow = sosfiltfilt(butter(2, 4, btype="low", fs=120, output="sos"), x, axis=-1)[:, :, 7::15].reshape(len(x), -1)
        self.wt = np.where(IU[0] == IU[1], 1.0, np.sqrt(2.0))

    def recentred(self, rows, reference):
        out, means = [], []
        for c in self.cov:
            cs = c[rows]
            mean = c[reference].mean(0)
            w = _isqrt(mean)[None]
            out.append(w @ cs @ w)
            means.append(mean)
        return out, means

    def tang(self, cw):
        return np.concatenate([_logm(c)[:, IU[0], IU[1]] * self.wt for c in cw], 1)

    def person(self, sessions, references=None):
        """sessions: row arrays of one person -> dict of own_* arrays (without the person axis)."""
        references = sessions if references is None else references
        cw, ms = zip(*[self.recentred(rows, ref) for rows, ref in zip(sessions, references, strict=True)])
        yt = np.concatenate([self.y[r] for r in sessions])
        prior = [np.mean([m[b] for m in ms], 0) for b in range(len(BANDS))]
        cws = [np.concatenate([c[b] for c in cw]) for b in range(len(BANDS))]
        tangent = make_pipeline(StandardScaler(), LogisticRegression(C=0.01, max_iter=500)).fit(self.tang(cws), yt)
        csp = []
        for c in cws:
            w = [eigh(c[yt == a].mean(0), c[yt == a].mean(0) + c[yt == b].mean(0))[1][:, [0, 1, 2, -3, -2, -1]]
                 for a, b in ((0, 1), (0, 2), (1, 2))]
            csp.append(np.concatenate(w, 1))
        feats = np.concatenate([np.log(np.einsum("cf,ncd,df->nf", w, c, w)) for w, c in zip(csp, cws)], 1)
        cls = LDA(solver="lsqr", shrinkage=0.5).fit(feats, yt)
        slow = np.concatenate([(self.slow[r] - self.slow[ref].mean(0)) / (self.slow[ref].std(0) + 1e-9)
                               for r, ref in zip(sessions, references, strict=True)])
        sl = LDA(solver="lsqr", shrinkage=0.7).fit(slow, yt)
        sc, lr = tangent.steps[0][1], tangent.steps[1][1]
        return dict(prior=np.stack(prior), csp=np.stack(csp), t_mu=sc.mean_, t_sd=sc.scale_, t_W=lr.coef_,
                    t_c=lr.intercept_, c_W=cls.coef_, c_c=cls.intercept_, s_W=sl.coef_, s_c=sl.intercept_)
