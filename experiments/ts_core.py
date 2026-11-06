"""Causal tangent-space core sweep (scratch harness, outside the repo).

Config name: <ref>_<bands>_<cov>_<est>_<refm>_k<k>[_st]
  ref   m1 (as exported) | car (common average reference)
  bands fb7 (v2) | fb12 | nb14 | nb18 | any BANDSETS key
  cov   full | sub2..4 (sub-window blocks) | fs2/fs3 (full + sub-windows) | ow3/fo3 (overlapping halves) | aug<L> (order-2 delay
        embedding, lag L samples) | augq (quarter-period lag per band)
  est   t<s> (trace shrinkage s; v2 t1e-3) | oas | lw
  refm  le (log-Euclidean, v2) | airm (Riemannian) | euc (arithmetic)
  k     weight (pseudo-windows) of the fold's training reference in the causal running reference
  st    per-session dispersion stretch (causal at test, prior = training dispersion)
Heads: lr<C>[ns] (scaler+LR; ns = no slow block), lda[ns], svm<C>, pb<C> (per-block LR, mean log-prob).
Training sessions use their whole-session reference (as v2); a test session's window t uses windows
0..t of that session plus the fold's training reference (k pseudo-windows); slow block centred the
same way (k zero windows). Folds and scores via experiments/protocol.py (evaluate).
"""
import argparse, json, os, re, sys, time
ROOT = os.environ.get("NI26_ROOT", "/Users/sebastianboehler/Documents/GitHub/neural-interfaces-2026")
sys.path[:0] = [ROOT + "/experiments"]
import numpy as np
from joblib import Parallel, delayed
from pyriemann.utils.geodesic import geodesic_riemann
from pyriemann.utils.mean import mean_riemann
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
import protocol
from features import bandpass, causal_references, expm_sym, logm, slow_waveform, whiten_tangent

SF = 120.0
BANDSETS = {
    "fb7": [(1, 4), (4, 8), (8, 13), (13, 20), (20, 30), (30, 45), (45, 58)],
    "fb12": [(1, 4), (4, 8), (8, 10), (10, 13), (13, 16), (16, 20), (20, 24), (24, 28), (28, 32),
             (32, 38), (38, 45), (45, 58)],
    "nb14": [(1, 4), (4, 6), (6, 8), (8, 10), (10, 12), (12, 14), (14, 17), (17, 20), (20, 23),
             (23, 26), (26, 30), (30, 34), (34, 40), (40, 58)],
    "nb18": [(1, 2.5), (2.5, 4), (4, 6), (6, 8), (8, 10), (10, 12), (12, 14), (14, 16), (16, 18),
             (18, 20), (20, 23), (23, 26), (26, 29), (29, 32), (32, 36), (36, 40), (40, 45), (45, 58)],
    "nb24": [(1, 2), (2, 3), (3, 4), (4, 5.5), (5.5, 7), (7, 8.5), (8.5, 10), (10, 11.5), (11.5, 13),
             (13, 14.5), (14.5, 16), (16, 18), (18, 20), (20, 22), (22, 24.5), (24.5, 27), (27, 30),
             (30, 33), (33, 36.5), (36.5, 40), (40, 45), (45, 50), (50, 55), (55, 58)],
    "ov17": [(1, 3), (2, 4), (3, 6), (4, 8), (6, 10), (8, 12), (10, 14), (12, 16), (14, 18), (16, 20),
             (18, 24), (20, 28), (24, 32), (28, 36), (32, 40), (36, 45), (40, 58)],
}
G1 = ("P01", "P02", "P05", "P07", "P09")


def parse(name):
    t = name.split("_")
    return dict(ref=t[0], bands=t[1], cov=t[2], est=t[3], refm=t[4], k=float(t[5][1:]), st="st" in t[6:])


def estimate(Xb, est, bw=None):
    X = Xb - Xb.mean(-1, keepdims=True)
    T, p = X.shape[-1], X.shape[1]
    Te = min(T, 2 * bw * T / SF) if est == "oase" else T  # oase: effective samples of a band
    S = np.einsum("nct,ndt->ncd", X, X, dtype=np.float64) / T
    mu = np.trace(S, axis1=1, axis2=2) / p
    F2 = (S ** 2).sum((1, 2))
    if est.startswith("t"):
        s = np.full(len(S), float(est[1:]))
    elif est in ("oas", "oase"):  # sklearn OAS formula (n_samples = Te)
        alpha = F2 / p ** 2
        s = np.clip((alpha + mu ** 2) / ((Te + 1) * (alpha - mu ** 2 / p)), 0, 1)
    elif est == "lw":
        n4 = (np.einsum("nct,nct->nt", X, X, dtype=np.float64) ** 2).sum(1)
        b2, d2 = (n4 - T * F2) / T ** 2, F2 - p * mu ** 2
        s = np.clip(np.minimum(b2, d2) / d2, 0, 1)
    return (1 - s)[:, None, None] * S + (s * mu)[:, None, None] * np.eye(p)


def blocks(X, cfg):
    out, cov = [], cfg["cov"]
    for lo, hi in BANDSETS[cfg["bands"]]:
        Xb = np.concatenate([bandpass(X[i:i + 256], lo, hi) for i in range(0, len(X), 256)])
        if cov == "full":
            segs = [Xb]
        elif cov[:3] == "sub" or cov[:2] == "fs":
            segs = ([Xb] if cov[:2] == "fs" else []) + np.array_split(Xb, int(cov[-1]), axis=-1)
        elif cov[:2] in ("ow", "fo"):  # 3 half-windows at 1/4 steps (+ full window for fo)
            T = Xb.shape[-1]
            segs = ([Xb] if cov[:2] == "fo" else []) + [Xb[..., i * T // 4:i * T // 4 + T // 2] for i in range(3)]
        elif cov[:3] == "aug":
            L = max(1, round(SF / (4 * np.sqrt(lo * hi)))) if cov == "augq" else int(cov[3:])
            segs = [np.concatenate([Xb[..., L:], Xb[..., :-L]], axis=1)]
        out += [estimate(s, cfg["est"], hi - lo) for s in segs]
    return out


class Feats:
    """Per-config covariance blocks, whole-session (training) features and causal test features."""

    def __init__(self, d, cfg):
        X = d["X"]
        if cfg["ref"] == "car":
            X = (X - X.mean(1, keepdims=True)).astype(X.dtype)
        self.cfg, self.k, self.st, self.refm, self.y = cfg, cfg["k"], cfg["st"], cfg["refm"], d["y"]
        self.C = blocks(X, cfg)
        self.L = [logm(C) for C in self.C] if self.refm == "le" else None
        st = d["stream"]
        rows = [np.flatnonzero(st == s) for s in dict.fromkeys(st)]
        self.M = slow_waveform(X, cutoff=1.0).astype(np.float64)
        F, self.d2, self.slices, a = [], [], [], 0
        for b, C in enumerate(self.C):
            V = np.zeros((len(C), C.shape[1] * (C.shape[1] + 1) // 2))
            for r in rows:
                V[r] = whiten_tangent(C[r], self.session_ref(b, r))
            d2 = (V ** 2).sum(1)
            if self.st:
                for r in rows:
                    V[r] /= np.sqrt(d2[r].mean())
            F.append(V), self.d2.append(d2), self.slices.append(slice(a, a + V.shape[1]))
            a += V.shape[1]
        Mc = self.M.copy()
        for r in rows:
            Mc[r] -= self.M[r].mean(0)
        F.append(Mc), self.slices.append(slice(a, a + Mc.shape[1]))
        self.F = np.concatenate(F, 1)

    def session_ref(self, b, r):
        if self.refm == "le":
            return expm_sym(self.L[b][r].mean(0))
        if self.refm == "airm":
            return mean_riemann(self.C[b][r])
        return self.C[b][r].mean(0)

    def prior(self, b, idx):  # fold training reference (log-domain for le)
        if self.refm == "le":
            return self.L[b][idx].mean(0)
        if self.refm == "airm":
            return mean_riemann(self.C[b][idx], init=expm_sym(logm(self.C[b][idx]).mean(0)))
        return self.C[b][idx].mean(0)

    def causal_refs(self, b, r, prior):
        C, k = self.C[b][r], self.k
        if self.refm == "le":
            return causal_references(self.L[b][r], prior, k)
        if self.refm == "euc":
            n = np.arange(1, len(r) + 1)[:, None, None]
            return (k * prior + np.cumsum(C, 0)) / (k + n)
        out, M = np.empty_like(C), prior
        for t in range(len(C)):
            M = out[t] = geodesic_riemann(M, C[t], 1.0 / (k + t + 1))
        return out

    def causal(self, r, priors, d2p):
        out, n = [], np.arange(1, len(r) + 1)
        for b in range(len(self.C)):
            V = whiten_tangent(self.C[b][r], self.causal_refs(b, r, priors[b]))
            if self.st:
                V = V / np.sqrt((self.k * d2p[b] + np.cumsum((V ** 2).sum(1))) / (self.k + n))[:, None]
            out.append(V)
        M = self.M[r]
        out.append(M - np.cumsum(M, 0) / (self.k + n[:, None]))
        return np.concatenate(out, 1)


def spec(h):
    kind, C, ns = re.fullmatch(r"(lr|svm|pb|lda)([\d.e-]*?)(ns)?", h).groups()
    return kind, float(C) if C else 0.0, bool(ns)


def softmax(z):
    z = z - z.max(1, keepdims=True)
    return np.exp(z) / np.exp(z).sum(1, keepdims=True)


def lr(C):
    return make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=2000))


def fit_head(h, X, y, sl):
    kind, C, ns = spec(h)
    X = X[:, :sl[-1].start] if ns else X
    if kind == "lr":
        return lr(C).fit(X, y)
    if kind == "svm":
        return make_pipeline(StandardScaler(), LinearSVC(C=C, max_iter=5000)).fit(X, y)
    if kind == "lda":
        return LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto").fit(X, y)
    return [lr(C).fit(X[:, s], y) for s in sl]


def predict_head(h, m, X, sl):
    kind, C, ns = spec(h)
    X = X[:, :sl[-1].start] if ns else X
    if kind == "svm":
        return softmax(m.decision_function(X))
    if kind == "pb":
        return softmax(np.mean([np.log(np.clip(mm.predict_proba(X[:, s]), 1e-9, 1))
                                for mm, s in zip(m, sl)], 0))
    return m.predict_proba(X)


class Core:
    """protocol.evaluate model: fit(train_idx); predict_proba_stream(one session, in order)."""

    def __init__(self, fe, heads, store):
        self.fe, self.heads, self.store = fe, heads, store

    def fit(self, idx):
        fe = self.fe
        self.priors = [fe.prior(b, idx) for b in range(len(fe.C))]
        self.d2p = [d2[idx].mean() for d2 in fe.d2]
        X, y = fe.F[idx], fe.y[idx]
        self.models = Parallel(n_jobs=min(4, len(self.heads)), prefer="threads")(
            delayed(fit_head)(h, X, y, fe.slices) for h in self.heads)

    def predict_proba_stream(self, idx):
        F = self.fe.causal(idx, self.priors, self.d2p)
        for h, m in zip(self.heads, self.models):
            self.store[h][idx] = predict_head(h, m, F, self.fe.slices)
        return self.store[self.heads[0]][idx]


def run(d, name, heads, out):
    t0 = time.time()
    fe = Feats(d, parse(name))
    heads = [h for h in heads if not (h.startswith("lda") and fe.F.shape[1] > 12000)]
    res, probs = {h: {} for h in heads}, {}
    for proto in ("all_loso", "seen_session"):
        store = {h: np.full((len(d["y"]), 3), np.nan) for h in heads}
        r = protocol.evaluate(lambda: Core(fe, heads, store), d, proto)
        assert np.allclose(store[heads[0]][r["index"]], r["proba"])
        te = np.zeros(len(d["y"]), bool)
        te[r["index"]] = True
        for h in heads:
            P = store[h]
            for p in sorted(set(d["subject"][te])):
                m = te & (d["subject"] == p)
                res[h].setdefault(proto, {})[p[-3:]] = protocol.score(
                    d["y"][m], P[m].argmax(1), d, m)["balanced_accuracy_cell_mean"]
            probs[f"{h}|{proto}"] = P[r["index"]].astype(np.float32)
        probs[f"index|{proto}"] = r["index"]
    secs = time.time() - t0
    with open(f"{out}/{name}.json", "w") as f:
        json.dump({"cfg": name, "dim": int(fe.F.shape[1]), "heads": res, "secs": secs}, f)
    np.savez_compressed(f"{out}/{name}.npz", **probs)
    for h in heads:
        a, s = res[h]["all_loso"], res[h]["seen_session"]
        g1 = np.mean([a[p] for p in G1]); g2 = np.mean([v for p, v in a.items() if p not in G1])
        print(f"{name} {h:9s} PE {100*np.mean(list(a.values())):.2f} G1 {100*g1:.2f} G2 {100*g2:.2f} "
              f"seen {100*np.mean(list(s.values())):.2f} dim {fe.F.shape[1]} {secs:.0f}s", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfgs", required=True)
    ap.add_argument("--heads", default="lr0.1,lr0.01,lr0.03,lr0.3")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    d = protocol.load()
    for name in a.cfgs.split(","):
        if os.path.exists(f"{a.out}/{name}.json"):
            continue
        run(d, name, a.heads.split(","), a.out)
