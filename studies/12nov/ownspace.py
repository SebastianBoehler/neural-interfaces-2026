"""Personal members in the person's own space (ledger s37): fitted on public sessions, streamed causally over a later one.
T: tangent space of 9 bands after recentring; C: FBCSP on the recentred covariances; S: slow time courses."""
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
Xc = X - X.mean(1, keepdims=True)
BANDS = [(1, 4), (4, 8), (8, 13), (13, 20), (20, 30), (30, 40), (40, 58), (4, 40), (8, 30)]
IU = np.triu_indices(41); WT = np.where(IU[0] == IU[1], 1.0, np.sqrt(2.0)); KREF = 2.0


def _cov(lo, hi):
    B = sosfiltfilt(butter(4, [lo, hi], btype="band", fs=120, output="sos"), Xc, axis=-1)
    C = np.einsum("nct,ndt->ncd", B, B) / B.shape[-1]
    return C + 1e-3 * np.trace(C, axis1=1, axis2=2)[:, None, None] / 41 * np.eye(41)


CV = [_cov(*b) for b in BANDS]
SL = sosfiltfilt(butter(2, 4, btype="low", fs=120, output="sos"), X, axis=-1)[:, :, 7::15].reshape(len(X), -1)


def _isqrt(M):
    w, V = np.linalg.eigh(M); return np.einsum("...ij,...j,...kj->...ik", V, 1 / np.sqrt(w), V)


def _logm(C):
    w, V = np.linalg.eigh(C); return np.einsum("...ij,...j,...kj->...ik", V, np.log(w), V)


def recentred(rows, prior=None):
    """-> per band whitened covariances (n,41,41). prior None: whole-session reference (training);
    else causal running reference with the prior mean as KREF pseudo-windows. Returns (list, list of session means)."""
    out, means = [], []
    for b, C in enumerate(CV):
        Cs = C[rows]
        if prior is None:
            M = Cs.mean(0); W = _isqrt(M)[None]
        else:
            n = np.arange(1, len(rows) + 1)[:, None, None]; M = (KREF * prior[b] + np.cumsum(Cs, 0)) / (KREF + n); W = _isqrt(M)
        out.append(W @ Cs @ W); means.append(Cs.mean(0))
    return out, means


def tang(Cw):
    return np.concatenate([(_logm(C)[:, IU[0], IU[1]]) * WT for C in Cw], 1)


def causal_z(F):
    n = np.arange(1, len(F) + 1)[:, None]; m = np.cumsum(F, 0) / n; v = np.cumsum(F ** 2, 0) / n - m ** 2
    return np.where(n >= 4, (F - m) / np.sqrt(np.maximum(v, 1e-9)), 0.0)


class Personal:
    def __init__(self, train_sessions):
        """train_sessions: list of row arrays, one per public session of the person."""
        cw, ms = zip(*[recentred(r) for r in train_sessions]); yt = np.concatenate([y[r] for r in train_sessions])
        self.prior = [np.mean([m[b] for m in ms], 0) for b in range(len(BANDS))]
        Cw = [np.concatenate([c[b] for c in cw]) for b in range(len(BANDS))]
        self.T = make_pipeline(StandardScaler(), LogisticRegression(C=0.01, max_iter=500)).fit(tang(Cw), yt)
        self.W = []
        for C in Cw:
            W = [eigh(C[yt == a].mean(0), C[yt == a].mean(0) + C[yt == b].mean(0))[1][:, [0, 1, 2, -3, -2, -1]] for a, b in ((0, 1), (0, 2), (1, 2))]
            self.W.append(np.concatenate(W, 1))
        self.C = LDA(solver="lsqr", shrinkage=0.5).fit(self._csp(Cw), yt)
        S = np.concatenate([(SL[r] - SL[r].mean(0)) / (SL[r].std(0) + 1e-9) for r in train_sessions])
        self.S = LDA(solver="lsqr", shrinkage=0.7).fit(S, yt)

    def _csp(self, Cw):
        return np.concatenate([np.log(np.einsum("cf,ncd,df->nf", W, C, W)) for W, C in zip(self.W, Cw)], 1)

    def stream(self, rows):
        """causal posteriors over one later session -> dict T, C, S (n,3)."""
        Cw, _ = recentred(rows, self.prior)
        return dict(T=self.T.predict_proba(tang(Cw)), C=self.C.predict_proba(self._csp(Cw)), S=self.S.predict_proba(causal_z(SL[rows])))


def ba(rows, P):
    p = P.argmax(1)
    return 100 * np.mean([np.mean([np.mean(p[(itf[rows] == i) & (y[rows] == k)] == k) for k in range(3)]) for i in np.unique(itf[rows])])
