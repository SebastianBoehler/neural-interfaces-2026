"""Routing v2 (route2): robust standardisation + shape score (population / debias directions projected out).
v1 score: cos(c - b - pop, G_p - pop) -> a session with weak or no class contrast lands on -(b + pop), which is close to
the weakest gallery (P04): 'weakness' was part of the fingerprint. v2 score: cos(P c, P G_p), P = projector onto the
orthogonal complement of span{pop, b} (768-dim vectors) -> invariant to contrast strength along the shared directions;
a session with no contrast scores 0. Standardisation: per-feature median / (1.4826 MAD), clipped at +-CLIP (artifacts)."""
import sys
import numpy as np
from sklearn.linear_model import LogisticRegression
import paths
S = paths.DATA
import rlib as L  # noqa: E402
import route_stream2 as RS  # noqa: E402

CLIP = 4.0


def rstats(X):
    med = np.median(X, 0); s = 1.4826 * np.median(np.abs(X - med), 0); return med, s


def rz(X, clip=CLIP, floor=1e-6):
    med, s = rstats(X); return np.clip((X - med) / np.maximum(s, floor), -clip, clip)


def contrast(X, W, npc=256, clip=CLIP):
    """Class means of robust-standardised rows (W: (n,3) 0/1 seed weights or one-hot labels) minus their average."""
    Z = rz(X[:, :npc], clip); n = W.sum(0)
    if (n < 1e-9).any():
        return None
    mu = (W.T @ Z) / n[:, None]; return mu - mu.mean(0)


def onehot(y):
    return (y[:, None] == np.arange(3)).astype(float)


def basis(pop, b):
    """Orthonormal basis (768, 2) of span{pop, b}."""
    return np.linalg.qr(np.stack([pop.ravel(), b.ravel()], 1))[0]


def perp(v, Q):
    v = v.ravel(); return v - Q @ (Q.T @ v)


def score(c, Gn, Q):
    """Shape scores (P,): cos(P c, P G_p); Gn = rows P G_p / |P G_p|. 0 if the projected probe is ~0."""
    a = perp(c, Q); na = np.linalg.norm(a)
    return Gn @ a / na if na > 1e-9 else np.zeros(len(Gn))


def gal_norm(gal, Q):
    G = np.stack([perp(g, Q) for g in gal]); return G / np.linalg.norm(G, axis=1, keepdims=True)


def causal_rz(F, clip=CLIP, t0=4):
    """Head input: row t robust-standardised over rows 0..t (same op as the stream); rows < t0 left at 0 (unused)."""
    out = np.zeros_like(F)
    for t in range(t0, len(F)):
        out[t] = RS.rz(F[:t + 1], F[t], clip, 1e-3)
    return out


def fold_model(proto, f, oof, npc=256, Cr=0.01):
    """Fold galleries (true-label robust contrasts, person mean), pop, debias b (q70 OOF - true), basis Q, heads."""
    tr, te = L.fold(proto, f); itr = np.load(f"{L.FE}/{proto}-{f}.npz")["idx_tr"]; st = L.STREAM[itr]
    per, bias = {}, {}
    for s, (X, y) in tr.items():
        per.setdefault(L.PER(s), []).append((s, X, y))
        Po = oof[itr[st == s]]
        bias.setdefault(L.PER(s), []).append(contrast(X, L.weights(Po, len(y), "q70", y), npc) - contrast(X, onehot(y), npc))
    names = list(per); gal = np.stack([np.mean([contrast(X, onehot(y), npc) for _, X, y in per[p]], 0) for p in names])
    pop = gal.mean(0); b = np.mean([np.mean(bias[p], 0) for p in names], 0); Q = basis(pop, b)
    heads = {}
    for p in names:
        Xh = np.concatenate([rz(X[:, :256]) for _, X, _ in per[p]])
        heads[p] = LogisticRegression(C=Cr, max_iter=3000).fit(Xh, np.concatenate([y for _, _, y in per[p]]))
    return te, names, gal, pop, b, Q, heads


def scores_session(X, T, Gn, Q, t_min=10, npc=256, rho=0.7, clip=CLIP):
    """Running v2 scores (n, P): row t from windows 0..t-1 with q70 seeds of T (NaN before t_min); stream ops."""
    out = np.full((len(X), len(Gn)), np.nan)
    for t in range(t_min, len(X)):
        c = RS.probe(X[:t, :npc], T[:t], rho, clip)
        if c is not None:
            out[t] = RS.shape_scores(c, Gn, Q)
    return out


def route_session(T, Sc, H, tau, ramp, wmax, t0):
    """Stream-identical routing given running scores Sc (n, P) and head probabilities H (P, n, 3)."""
    out = T.copy(); W = np.zeros(len(T)); who = np.full(len(T), -1)
    for t in range(t0, len(T)):
        if not np.isfinite(Sc[t]).all():
            continue
        j = int(np.argmax(Sc[t])); w = wmax * np.clip((Sc[t, j] - tau) / ramp, 0, 1)
        if w > 0:
            o = RS.blend(T[t], H[j][t], w)
            if np.isfinite(o).all():
                out[t] = o; W[t] = w; who[t] = j
    return out, W, who
