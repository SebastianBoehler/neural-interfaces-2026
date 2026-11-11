"""Class-direction fingerprint (route team): loaders + contrast scores on core1's exact-SVD PC view (feats_exact)."""
import numpy as np
import paths
S, REPO = paths.DATA, paths.REPO
FE = S + "/feats_exact"
with np.load(paths.WINDOWS) as f:
    STREAM = np.char.add(np.char.add(f["subject"], "|"), f["session"]); Y = f["y"]
PER = lambda s: s.split("|")[0].split("/")[-1]
SNO = lambda s: int(s.split("|")[1])
PEOPLE = [f"P{i:02d}" for i in range(1, 11)]


def fold(proto, name):
    """-> train {stream: (X float64, y)}, test {stream: (X, y, idx)} in acquisition order."""
    z = np.load(f"{FE}/{proto}-{name}.npz")
    st = STREAM[z["idx_tr"]]; Xtr = z["pc_tr"].astype(np.float64); tr = {}
    for s in dict.fromkeys(st):
        m = st == s; tr[s] = (Xtr[m], z["y_tr"][m])
    te = {}
    for k, s in enumerate(z["sessions"]):
        i = z[f"{k}|idx"]; te[str(s)] = (z[f"{k}|pc"].astype(np.float64), Y[i], i)
    return tr, te


def contrast(X, y, w=None, npc=256, std=True, Wh=None):
    """Per-session standardised class means minus their average -> (3, d); w = per-row weights (seed mask/soft)."""
    X = X[:, :npc]; m = X.mean(0); Z = X - m
    if std:
        Z = Z / np.maximum(X.std(0), 1e-6)
    if Wh is not None:
        Z = Z @ Wh
    if w is None or w.ndim == 1:
        w = np.ones(len(X)) if w is None else w
        W = w[:, None] * (y[:, None] == np.arange(3))
    else:
        W = w  # (n, 3) soft class weights
    mu = []
    for k in range(3):
        a = W[:, k]
        if a.sum() < 1e-9:
            return None
        mu.append((a[:, None] * Z).sum(0) / a.sum())
    mu = np.stack(mu); return mu - mu.mean(0)


def whitener(tr, npc, shrink=0.5):
    """Pooled within-class covariance of per-session standardised training rows -> W (d, d) = C^-1/2 (shrunk)."""
    R = []
    for X, y in tr.values():
        X = X[:, :npc]; Z = (X - X.mean(0)) / np.maximum(X.std(0), 1e-6)
        for k in range(3):
            R.append(Z[y == k] - Z[y == k].mean(0))
    R = np.concatenate(R); C = np.cov(R.T); C = (1 - shrink) * C + shrink * np.eye(npc) * np.trace(C) / npc
    w, V = np.linalg.eigh(C); return (V / np.sqrt(w)) @ V.T


def cos(a, b):
    a, b = a.ravel(), b.ravel(); return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def subspace(a, b):
    """Mean squared cosine of the principal angles between the 2-dim contrast subspaces."""
    qa = np.linalg.svd(a.T, full_matrices=False)[0][:, :2]; qb = np.linalg.svd(b.T, full_matrices=False)[0][:, :2]
    return float((np.linalg.svd(qa.T @ qb, compute_uv=False) ** 2).mean())


def score(cp, cg, pop, kind):
    if kind == "cos":
        return cos(cp, cg)
    if kind == "res":
        return cos(cp - pop, cg - pop)
    if kind == "sub":
        return subspace(cp, cg)
    if kind == "subres":
        return subspace(cp - pop, cg - pop)
    raise ValueError(kind)


def weights(P, t, src, y=None, rho=0.7):
    """Class weights (t, 3) of windows 0..t-1 from posteriors P (true labels for src='true')."""
    if src == "true":
        return (y[:t, None] == np.arange(3)).astype(float)
    Pt = P[:t]; lab = Pt.argmax(1); conf = Pt.max(1)
    if src == "soft":
        return Pt
    W = np.zeros((t, 3))
    if src == "hard":
        W[np.arange(t), lab] = 1; return W
    q = max(2, int(rho * t / 3))
    for k in range(3):
        ik = np.flatnonzero(lab == k); W[ik[np.argsort(-conf[ik])[:q]], k] = 1
    return W
