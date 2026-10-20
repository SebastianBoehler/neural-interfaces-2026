"""xDAWN / ERP-covariance decoder on the slow, session-centred waveform.

Each window is low-passed (``cutoff`` Hz), decimated and centred by its
session's mean waveform (training: whole session; test: causal running
mean with ``k`` zero pseudo-windows). xDAWN spatial filters (``nf`` per
class) and class templates come from the training windows only. Every
window becomes a super-trial [filtered templates; filtered window], whose
covariance is recentred by its session's causal log-Euclidean mean (as in
``TangentSpace``) and mapped to the tangent space; logistic regression.
"""

import numpy as np
from scipy.linalg import eigh
from scipy.signal import butter, sosfiltfilt

from features import SFREQ, causal_references, covariances, expm_sym, logm, whiten_tangent
from models import make_clf


class ErpTangent:
    _slow = {}

    def __init__(self, d, cutoff=4.0, step=4, nf=4, k=5.0, C=0.1, shrink=1e-2,
                 align=True):
        self.d, self.nf, self.k, self.shrink, self.align = d, nf, k, shrink, align
        key = (id(d["X"]), cutoff, step)
        if key not in self._slow:
            sos = butter(4, cutoff, btype="low", fs=SFREQ, output="sos")
            self._slow[key] = np.concatenate(
                [sosfiltfilt(sos, d["X"][i:i + 256], axis=-1)[..., ::step]
                 for i in range(0, len(d["X"]), 256)]).astype(np.float64)
        self.S = self._slow[key]
        self.clf = make_clf("lr", C)

    def _centred(self, idx, causal):
        S = self.S[idx]
        if not causal:
            return S - S.mean(0)
        n = np.arange(1, len(idx) + 1)[:, None, None]
        return S - np.cumsum(S, 0) / (self.k + n)

    def _covs(self, Z):
        T = np.broadcast_to(self.T, (len(Z),) + self.T.shape)
        return covariances(np.concatenate([T, np.einsum("fc,nct->nft", self.W, Z)], 1),
                           self.shrink)

    def _tangent(self, idx, causal):
        Cv = self._covs(self._centred(idx, causal))
        if not self.align:
            return whiten_tangent(Cv, self.glob)
        L = logm(Cv)
        R = (causal_references(L, self.log_glob, self.k) if causal
             else expm_sym(L.mean(0)))
        return whiten_tangent(Cv, R)

    def fit(self, idx):
        st, y = self.d["stream"][idx], self.d["y"][idx]
        Z = np.concatenate([self._centred(idx[st == s], False) for s in dict.fromkeys(st)])
        y = np.concatenate([y[st == s] for s in dict.fromkeys(st)])
        Ctot = np.einsum("nct,ndt->cd", Z, Z) / (len(Z) * Z.shape[-1])
        Ctot += 1e-3 * np.trace(Ctot) / len(Ctot) * np.eye(len(Ctot))
        W, T = [], []
        for c in range(3):
            P = Z[y == c].mean(0)
            _, V = eigh(P @ P.T / P.shape[-1], Ctot)
            Wc = V[:, ::-1][:, :self.nf].T
            W.append(Wc)
            T.append(Wc @ P)
        self.W, self.T = np.concatenate(W), np.concatenate(T)
        Cv = self._covs(Z)
        self.log_glob = logm(Cv).mean(0)
        self.glob = expm_sym(self.log_glob)
        F = np.concatenate([self._tangent(idx[st == s], False) for s in dict.fromkeys(st)])
        self.clf.fit(F, y)

    def predict_proba_stream(self, idx):
        return self.clf.predict_proba(self._tangent(idx, True))

    def predict_stream(self, idx):
        return self.predict_proba_stream(idx).argmax(1)
