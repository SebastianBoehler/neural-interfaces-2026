"""v6.6: capped causal personalisation of unrecognised sessions (ledger s260 form, s268 package; study
studies/14nov/causalpers: fast.py a2, run.py cq50, score260.py fusion). It reads only the window, the package's emitted
posterior p (before the count filter) and the count filter's output o, and changes only the label predict() emits.
State lives on the model and is reset with the session; numpy / scipy / sklearn / pyriemann on the CPU, no clock.

Per session, in delivery order, until the first routed (recognised) window:
- features of every window: zero-phase band covariance (OAS) in each of the eight FBCSP bands;
- refit at window t = t0, t0 + k, ...: the chain labels are argmax p of windows < t; per chain class the top share q of
  windows by confidence (at least 3) trains one-vs-rest CSP (2 filter pairs per class and band, log-variance features)
  and an unweighted shrinkage LDA; fewer than two chain classes: uniform;
- window i >= t0 takes the posterior q of the latest refit; output = log o + w(i) log q with w(i) = cap clip((i - w0 + 1) / R, 0, 1),
  nothing is changed while w(i) = 0.
Any exception: counted, one stderr line, o for the rest of the session. After recognition the stage is stopped for good."""
import sys

import numpy as np
from scipy.linalg import eigh
from scipy.signal import butter, sosfiltfilt


class CausalPers:
    def __init__(self, z):
        from pyriemann.estimation import Covariances  # imported at load: a missing library fails the package check, not a session
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
        self.cov, self.LDA = Covariances("oas"), LinearDiscriminantAnalysis
        fs = float(z["v6_cp_fs"])
        self.sos = [butter(4, tuple(b), "bandpass", fs=fs, output="sos") for b in np.asarray(z["v6_cp_bands"], np.float64)]
        self.t0, self.k, self.w0, self.R = (int(z[f"v6_cp_{n}"]) for n in ("t0", "k", "w0", "R"))
        self.cap, self.q = float(z["v6_cp_cap"]), float(z["v6_cp_q"])
        self.reset_state()

    def reset_state(self):
        self.report()
        self.t, self.C, self.P, self.model = 0, [], [], None
        self.stopped, self.dead, self.refits, self.fused, self.nfail = None, False, 0, 0, 0
        self.log = dict(err=0, first="")

    def report(self):
        log = getattr(self, "log", None)
        if log is not None and self.t:
            print(f"[cpers] windows {self.t} stopped_at {self.stopped} refits {self.refits} fused {self.fused} "
                  f"single_class {self.nfail} errors {log['err']}{' first: ' + log['first'] if log['first'] else ''}",
                  file=sys.stderr, flush=True)

    def __del__(self):  # the evaluator drops a session's copy without a call: report then (as CountFilter)
        try:
            self.report()
        except Exception:
            pass

    def feats(self, x):
        X = np.asarray(x, dtype=np.float64)
        return np.stack([self.cov.transform(sosfiltfilt(sos, X, axis=-1)[None])[0] for sos in self.sos])

    def select(self, y, c):
        keep = []
        for k in np.unique(y):
            r = np.flatnonzero(y == k)
            keep.append(r[np.argsort(-c[r])[:max(3, int(np.ceil(self.q * len(r))))]])
        return np.sort(np.concatenate(keep))

    def refit(self, t):
        """windows 0..t-1 -> (list of per-band CSP filters, LDA) or None (fewer than two chain classes)"""
        P = np.asarray(self.P[:t]); y, c = P.argmax(1), P.max(1)
        tr = self.select(y, c)
        if len(np.unique(y[tr])) < 2:
            self.nfail += 1
            return None
        C = np.stack(self.C[:t])  # (t, bands, 41, 41)
        Ws, A = [], []
        for b in range(len(self.sos)):
            Cb, W = C[tr, b], []
            for cl in np.unique(y[tr]):
                a, o = y[tr] == cl, y[tr] != cl
                Ma, Mo = np.average(Cb[a], 0, np.ones(a.sum())), np.average(Cb[o], 0, np.ones(o.sum()))
                Mo = 0.9 * Mo + 0.1 * np.trace(Mo) / len(Mo) * np.eye(len(Mo))
                _, V = eigh(Ma, Ma + Mo); W += [V[:, :2], V[:, -2:]]
            W = np.concatenate(W, 1); Ws.append(W); A.append(self.lv(W, Cb))
        lda = self.LDA(solver="lsqr", shrinkage="auto").fit(np.concatenate(A, 1), y[tr])
        return Ws, lda

    @staticmethod
    def lv(W, C):
        return np.log(np.einsum("ck,ncd,dk->nk", W, C, W))

    def predict(self, Cw):
        if self.model is None:
            return np.full(3, 1 / 3)
        Ws, lda = self.model
        f = np.concatenate([self.lv(W, Cw[None, b]) for b, W in enumerate(Ws)], 1)
        out = np.zeros(3)
        out[np.asarray(lda.classes_, int)] = lda.predict_proba(f)[0]
        return out

    def step(self, x, p, o, recognised):
        """x: the window as step() saw it; p: the package's posterior for it; o: the count filter's output; recognised: a
        window of this session has been routed. -> the posterior predict() takes the label from."""
        i = self.t
        self.t += 1
        if self.dead or self.stopped is not None:
            return o
        try:
            if recognised:
                self.stopped = i
                self.C, self.P, self.model = [], [], None
                return o
            self.C.append(self.feats(x)); self.P.append(np.asarray(p, np.float64))
            if i >= self.t0 and (i - self.t0) % self.k == 0:
                self.model = self.refit(i); self.refits += 1
            w = self.cap * min(max((i - self.w0 + 1) / self.R, 0.0), 1.0)
            if i < self.t0 or w <= 0:
                return o
            q = self.predict(self.C[i])
            L = np.log(np.clip(np.asarray(o, np.float64), 1e-6, 1)) + w * np.log(np.clip(q, 1e-6, 1))
            if not np.isfinite(L).all():
                raise FloatingPointError("personalised scores not finite")
            self.fused += 1
            return L
        except Exception as e:  # the rest of the session emits o
            self.dead = True
            try:
                self.log["err"] += 1
                self.log["first"] = self.log["first"] or repr(e)[:200]
                print(f"[cpers] error at window {i}: {repr(e)[:200]}; count-filter output for the rest of the session",
                      file=sys.stderr, flush=True)
            except Exception:
                pass
            return o
