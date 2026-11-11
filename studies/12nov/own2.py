"""Own-space personal members, configurable (ledger s44). Same members as ownspace.py at the default config."""
import os, warnings
import numpy as np
from scipy.signal import butter, sosfiltfilt
from scipy.linalg import eigh
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
warnings.filterwarnings("ignore")
REPO = os.path.abspath(__file__ + "/../../..")
z = np.load(REPO + "/data/proteus_windows.npz", allow_pickle=True)
X, y, itf = z["X"].astype(np.float64), z["y"], z["interface"]
PER = np.array([s[-3:] for s in z["subject"]]); SES = z["session"].astype(str)
RUN = np.char.add(z["task"].astype(str), z["run"].astype(str))
Xc = X - X.mean(1, keepdims=True)
B9 = ((1, 4), (4, 8), (8, 13), (13, 20), (20, 30), (30, 40), (40, 58), (4, 40), (8, 30))
OV17 = ((1, 3), (2, 4), (3, 6), (4, 8), (6, 10), (8, 12), (10, 14), (12, 16), (14, 18), (16, 20), (18, 24), (20, 28), (24, 32), (28, 36),
        (32, 40), (36, 45), (40, 58))
IU = np.triu_indices(41); WT = np.where(IU[0] == IU[1], 1.0, np.sqrt(2.0)); KREF = 2.0
DEF = dict(bands=B9, halves=False, csp_k=3, shr_c=0.5, lr_c=0.01, calib=False)
_CV = {}
SL = sosfiltfilt(butter(2, 4, btype="low", fs=120, output="sos"), X, axis=-1)[:, :, 7::15].reshape(len(X), -1)
TEMPS = np.exp(np.linspace(np.log(0.2), np.log(8), 25))


def cov(band, seg=None):
    k = (band, seg)
    if k not in _CV:
        B = sosfiltfilt(butter(4, list(band), btype="band", fs=120, output="sos"), Xc, axis=-1)
        B = B if seg is None else B[:, :, seg[0]:seg[1]]
        C = np.einsum("nct,ndt->ncd", B, B) / B.shape[-1]; _CV[k] = C + 1e-3 * np.trace(C, axis1=1, axis2=2)[:, None, None] / 41 * np.eye(41)
    return _CV[k]


def _isqrt(M):
    w, V = np.linalg.eigh(M); return np.einsum("...ij,...j,...kj->...ik", V, 1 / np.sqrt(w), V)


def _logm(C):
    w, V = np.linalg.eigh(C); return np.einsum("...ij,...j,...kj->...ik", V, np.log(w), V)


def causal_z(F):
    n = np.arange(1, len(F) + 1)[:, None]; m = np.cumsum(F, 0) / n; v = np.cumsum(F ** 2, 0) / n - m ** 2
    return np.where(n >= 4, (F - m) / np.sqrt(np.maximum(v, 1e-9)), 0.0)


def temper(P, t):
    L = np.log(np.clip(P, 1e-6, 1)) / t; e = np.exp(L - L.max(1, keepdims=True)); return e / e.sum(1, keepdims=True)


class Personal:
    def __init__(self, train, cfg=None, _inner=False):
        self.cfg = c = dict(DEF, **(cfg or {})); self.blocks = [(b, None) for b in c["bands"]]
        self.tblocks = self.blocks + ([(b, s) for b in c["bands"] for s in ((0, 240), (240, 480))] if c["halves"] else [])
        yt = np.concatenate([y[r] for r in train]); self.prior = {k: np.mean([cov(*k)[r].mean(0) for r in train], 0) for k in self.tblocks}
        W = lambda k: np.concatenate([self._w(k, r, None) for r in train])  # noqa: E731
        self.T = make_pipeline(StandardScaler(), LogisticRegression(C=c["lr_c"], max_iter=500)).fit(self._tang(lambda k: W(k)), yt)
        Cw = {k: W(k) for k in self.blocks}; q = c["csp_k"]; sel = list(range(q)) + list(range(-q, 0)); self.F = {}
        for k in self.blocks:
            self.F[k] = np.concatenate([eigh(Cw[k][yt == a].mean(0), Cw[k][yt == a].mean(0) + Cw[k][yt == b].mean(0))[1][:, sel] for a, b in ((0, 1), (0, 2), (1, 2))], 1)
        self.C = LDA(solver="lsqr", shrinkage=c["shr_c"]).fit(self._csp(lambda k: Cw[k]), yt)
        self.S = LDA(solver="lsqr", shrinkage=0.7).fit(np.concatenate([(SL[r] - SL[r].mean(0)) / (SL[r].std(0) + 1e-9) for r in train]), yt)
        self.temp = dict(T=1.0, C=1.0, S=1.0)
        if c["calib"] and not _inner:
            allr = np.concatenate(train); runs = np.char.add(SES[allr], RUN[allr]); oof = {k: np.zeros((len(allr), 3)) for k in "TCS"}
            for u in dict.fromkeys(runs):
                te = allr[runs == u]; tr = [r[~np.isin(r, te)] for r in train]; o = Personal([r for r in tr if len(r)], cfg, _inner=True).stream(te)
                for k in "TCS":
                    oof[k][runs == u] = o[k]
            for k in "TCS":
                nll = [-np.mean(np.log(temper(oof[k], t)[np.arange(len(allr)), yt])) for t in TEMPS]; self.temp[k] = float(TEMPS[int(np.argmin(nll))])

    def _w(self, k, rows, causal):
        Cs = cov(*k)[rows]
        if causal is None:
            Wm = _isqrt(Cs.mean(0))[None]
        else:
            n = np.arange(1, len(rows) + 1)[:, None, None]; Wm = _isqrt((KREF * self.prior[k] + np.cumsum(Cs, 0)) / (KREF + n))
        return Wm @ Cs @ Wm

    def _tang(self, get):
        return np.concatenate([(_logm(get(k))[:, IU[0], IU[1]]) * WT for k in self.tblocks], 1)

    def _csp(self, get):
        return np.concatenate([np.log(np.einsum("cf,ncd,df->nf", self.F[k], get(k), self.F[k])) for k in self.blocks], 1)

    def stream(self, rows):
        Cw = {k: self._w(k, rows, True) for k in self.tblocks}
        o = dict(T=self.T.predict_proba(self._tang(lambda k: Cw[k])), C=self.C.predict_proba(self._csp(lambda k: Cw[k])), S=self.S.predict_proba(causal_z(SL[rows])))
        return {k: temper(v, self.temp[k]) for k, v in o.items()}


def ba(rows, P):
    p = P.argmax(1)
    return 100 * np.mean([np.mean([np.mean(p[(itf[rows] == i) & (y[rows] == k)] == k) for k in range(3)]) for i in np.unique(itf[rows])])
