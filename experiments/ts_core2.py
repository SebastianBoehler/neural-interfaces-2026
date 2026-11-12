"""Round-2 causal tangent-space core sweep (scratch harness; le/trace-shrinkage subset of ts_core.py, same numbers).

Config name: <ref>_<bands>_<cov>_t<s>_le_k<k>[_flags]
  ref   m1 | car | cm (CAR blocks + m1 blocks concatenated, both slow blocks)
  bands BANDSETS key, or a '+'-joined union (e.g. fb7+ov17)
  cov   [pre]<layout>; layout full | sub<n> | fs<n> (full + n sub-windows) | ow3 | fo3
        pre  a<L>: order-2 delay embedding, lag L samples (2C channels); aq: quarter-period lag per band;
             h: covariance of the band-limited analytic signal (complex Hermitian; tangent = Re upper + Im strict upper)
  flags ct / cb (train on causal-reference features of the training sessions / on both, fold prior)
        st (dispersion stretch) | sz<k> (per-session causal standardisation of the tangent features: training
        sessions z-scored with their own mean/var; a test window t uses windows 0..t plus the fold's mean
        session stats weighted as k pseudo-windows)
Heads: lr<C> (scaler + LR), mdm (per-block nearest class mean in the recentred tangent space).
Training sessions use their whole-session reference (as v2); test references are causal (windows 0..t + prior).
"""
import argparse, json, os, re, sys, time
ROOT = os.environ.get("NI26_ROOT", "/Users/sebastianboehler/Documents/GitHub/neural-interfaces-2026")
sys.path[:0] = [ROOT + "/experiments"]
import numpy as np
from joblib import Parallel, delayed
from scipy.signal import hilbert
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import protocol
import features as FT
from features import bandpass, slow_waveform

SF = 120.0
BANDSETS = {
    "fb7": [(1, 4), (4, 8), (8, 13), (13, 20), (20, 30), (30, 45), (45, 58)],
    "fb12": [(1, 4), (4, 8), (8, 10), (10, 13), (13, 16), (16, 20), (20, 24), (24, 28), (28, 32),
             (32, 38), (38, 45), (45, 58)],
    "ov17": [(1, 3), (2, 4), (3, 6), (4, 8), (6, 10), (8, 12), (10, 14), (12, 16), (14, 18), (16, 20),
             (18, 24), (20, 28), (24, 32), (28, 36), (32, 40), (36, 45), (40, 58)],
    # wider overlapping bands (about 1 octave, half-octave steps)
    "ow11": [(1, 4), (2, 6), (4, 8), (6, 12), (8, 16), (12, 20), (16, 26), (20, 32), (26, 40), (32, 48), (40, 58)],
    # denser constant-ratio bands (ratio 1.5, step 1.22)
    "ov22": [(1, 2.5), (1.5, 3), (2, 4), (3, 5), (4, 6), (5, 7.5), (6, 9), (7.5, 11), (9, 13), (11, 16), (13, 19),
             (16, 23), (19, 27), (23, 32), (27, 38), (32, 45), (38, 52), (45, 58), (8, 30), (1, 8), (30, 58), (4, 45)],
}
G1 = ("P01", "P02", "P05", "P07", "P09")


def parse(name):
    t = name.split("_")
    fl = t[6:]
    # audit of 12 Nov (ledger s52): unknown flags were ignored and the reference-mean token was never used
    assert all(f in ("st", "ct", "cb") or re.fullmatch(r"sz[0-9.]+", f) for f in fl), f"unknown flag in {name}: {fl}"
    assert t[4] == "le", f"only the log-Euclidean reference is implemented here, got {t[4]}"
    sz = next((float(f[2:]) for f in fl if f.startswith("sz")), 0.0)
    return dict(ref=t[0], bands=t[1], cov=t[2], est=t[3], refm=t[4], k=float(t[5][1:]), st="st" in fl, sz=sz,
                ct="ct" in fl, cb="cb" in fl)


# complex-safe SPD/HPD helpers; the real path calls features.py itself (bit-identical to round 1)
def _efn(C, fn):
    w, V = np.linalg.eigh(C)
    return np.einsum("...ij,...j,...kj->...ik", V, fn(w), V.conj())


def logm(C):
    return FT.logm(C) if not np.iscomplexobj(C) else _efn(C, lambda w: np.log(np.maximum(w, 1e-12)))


def expm(L):
    return FT.expm_sym(L) if not np.iscomplexobj(L) else _efn(L, np.exp)


def whiten_tangent(C, R):
    if not np.iscomplexobj(C):
        return FT.whiten_tangent(C, R)
    P = _efn(R, lambda w: 1.0 / np.sqrt(np.maximum(w, 1e-12)))
    W = P @ C @ P if P.ndim == 2 else np.einsum("nij,njk,nkl->nil", P, C, P)
    L = logm(W)
    n = L.shape[-1]
    iu = np.triu_indices(n, 1)
    return np.concatenate([FT.upper(L.real), np.sqrt(2.0) * L.imag[..., iu[0], iu[1]]], -1)


def estimate(Xb, est):
    X = Xb - Xb.mean(-1, keepdims=True)
    T, p = X.shape[-1], X.shape[1]
    cx = np.iscomplexobj(X)
    S = np.einsum("nct,ndt->ncd", X, X.conj() if cx else X, dtype=np.complex128 if cx else np.float64) / T
    mu = np.trace(S, axis1=1, axis2=2).real / p
    s = np.full(len(S), float(est[1:]))
    return (1 - s)[:, None, None] * S + (s * mu)[:, None, None] * np.eye(p)


def bandset(key):
    return [b for k in key.split("+") for b in BANDSETS[k]]


def blocks(X, cfg):
    out = []
    pre, lay = re.fullmatch(r"(h|aq|a\d+)?(full|sub\d|fs\d|ow3|fo3)", cfg["cov"]).groups()
    for lo, hi in bandset(cfg["bands"]):
        Xb = np.concatenate([bandpass(X[i:i + 256], lo, hi) for i in range(0, len(X), 256)])
        if pre == "h":
            Xb = hilbert(Xb, axis=-1).astype(np.complex64)
        elif pre:
            L = max(1, round(SF / (4 * np.sqrt(lo * hi)))) if pre == "aq" else int(pre[1:])
            Xb = np.concatenate([Xb[..., L:], Xb[..., :-L]], axis=1)
        T = Xb.shape[-1]
        if lay == "full":
            segs = [Xb]
        elif lay[:3] == "sub" or lay[:2] == "fs":
            segs = ([Xb] if lay[:2] == "fs" else []) + np.array_split(Xb, int(lay[-1]), axis=-1)
        else:  # ow3 / fo3: 3 half-windows at 1/4 steps (+ full window for fo)
            segs = ([Xb] if lay[:2] == "fo" else []) + [Xb[..., i * T // 4:i * T // 4 + T // 2] for i in range(3)]
        out += [estimate(s, cfg["est"]) for s in segs]
    return out


class Feats:
    """Per-config covariance blocks, whole-session (training) features and causal test features."""

    def __init__(self, d, cfg):
        X0 = d["X"]
        refs = {"m1": ["m1"], "car": ["car"], "cm": ["car", "m1"]}[cfg["ref"]]
        self.cfg, self.k, self.st, self.sz, self.y = cfg, cfg["k"], cfg["st"], cfg["sz"], d["y"]
        self.C, Ms = [], []
        for rf in refs:
            X = (X0 - X0.mean(1, keepdims=True)).astype(X0.dtype) if rf == "car" else X0
            self.C += blocks(X, cfg)
            Ms.append(slow_waveform(X, cutoff=1.0).astype(np.float64))
        self.M = np.concatenate(Ms, 1)
        self.L = [logm(C) for C in self.C]
        st = d["stream"]
        self.sess = list(dict.fromkeys(st))
        rows = [np.flatnonzero(st == s) for s in self.sess]
        self.rows = rows
        F, self.d2, self.slices, a = [], [], [], 0
        for b, C in enumerate(self.C):
            V = None
            for r in rows:
                v = whiten_tangent(C[r], expm(self.L[b][r].mean(0)))
                if V is None:
                    V = np.zeros((len(C), v.shape[1]))
                V[r] = v
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
        del F
        if self.sz:  # per-session z-scoring; session stats kept for the fold prior
            self.mu = np.stack([self.F[r].mean(0) for r in rows])
            self.e2 = np.stack([(self.F[r] ** 2).mean(0) for r in rows])
            for i, r in enumerate(rows):
                self.F[r] = (self.F[r] - self.mu[i]) / np.sqrt(np.maximum(self.e2[i] - self.mu[i] ** 2, 1e-12))

    def prior(self, b, idx):
        return self.L[b][idx].mean(0)

    def sz_prior(self, idx):
        if not self.sz:
            return None
        inn = np.zeros(len(self.y), bool)
        inn[idx] = True
        ss = [i for i, r in enumerate(self.rows) if inn[r].any()]
        return self.mu[ss].mean(0), self.e2[ss].mean(0)

    def causal(self, r, priors, d2p, szp):
        out, n = [], np.arange(1, len(r) + 1)
        for b in range(len(self.C)):
            V = whiten_tangent(self.C[b][r], expm((self.k * priors[b] + np.cumsum(self.L[b][r], 0))
                                                  / (self.k + n)[:, None, None]))
            if self.st:
                V = V / np.sqrt((self.k * d2p[b] + np.cumsum((V ** 2).sum(1))) / (self.k + n))[:, None]
            out.append(V)
        M = self.M[r]
        out.append(M - np.cumsum(M, 0) / (self.k + n[:, None]))
        F = np.concatenate(out, 1)
        if self.sz:
            mu0, e20 = szp
            kk, nn = self.sz, (self.sz + n)[:, None]
            m = (kk * mu0 + np.cumsum(F, 0)) / nn
            e2 = (kk * e20 + np.cumsum(F ** 2, 0)) / nn
            F = (F - m) / np.sqrt(np.maximum(e2 - m ** 2, 1e-12))
        return F


def softmax(z):
    z = z - z.max(1, keepdims=True)
    return np.exp(z) / np.exp(z).sum(1, keepdims=True)


def fit_head(h, X, y, sl):
    if h == "mdm":
        cls = np.unique(y)
        mus = np.stack([X[y == c].mean(0) for c in cls])
        dd = np.stack([((X - mus[i]) ** 2) for i in range(len(cls))])  # (K, n, D)
        within = np.mean([dd[i][y == c].sum(1).mean() for i, c in enumerate(cls)])
        return mus, within / len(sl)
    return make_pipeline(StandardScaler(), LogisticRegression(C=float(h[2:]), max_iter=2000)).fit(X, y)


def predict_head(h, m, X, sl):
    if h == "mdm":
        mus, sc = m
        D = np.stack([((X - mu) ** 2).sum(1) for mu in mus], 1)
        return softmax(-D / sc)
    return m.predict_proba(X)


class Core:
    """protocol.evaluate model: fit(train_idx); predict_proba_stream(one session, in order)."""

    def __init__(self, fe, heads, store):
        self.fe, self.heads, self.store = fe, heads, store

    def fit(self, idx):
        fe = self.fe
        self.priors = [fe.prior(b, idx) for b in range(len(fe.C))]
        self.d2p = [d2[idx].mean() for d2 in fe.d2]
        self.szp = fe.sz_prior(idx)
        X, y = fe.F[idx], fe.y[idx]
        if fe.cfg["ct"] or fe.cfg["cb"]:  # training features with the test-time causal references (fold prior)
            rs = [r[np.isin(r, idx)] for r in fe.rows if np.isin(r, idx).any()]
            Xc = np.concatenate([fe.causal(r, self.priors, self.d2p, self.szp) for r in rs])
            yc = fe.y[np.concatenate(rs)]
            X, y = (Xc, yc) if fe.cfg["ct"] else (np.concatenate([X, Xc]), np.concatenate([y, yc]))
        self.models = Parallel(n_jobs=min(4, len(self.heads)), prefer="threads")(
            delayed(fit_head)(h, X, y, fe.slices) for h in self.heads)

    def predict_proba_stream(self, idx):
        F = self.fe.causal(idx, self.priors, self.d2p, self.szp)
        for h, m in zip(self.heads, self.models):
            self.store[h][idx] = predict_head(h, m, F, self.fe.slices)
        return self.store[self.heads[0]][idx]


def run(d, name, heads, out):
    t0 = time.time()
    fe = Feats(d, parse(name))
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
    ap.add_argument("--heads", default="lr3,lr10")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    d = protocol.load()
    for name in a.cfgs.split(","):
        if os.path.exists(f"{a.out}/{name}.json"):
            continue
        run(d, name, a.heads.split(","), a.out)
