"""Class-direction routing (route team): causal soft routing of B2+TSEM output to a returning person's personal head.
Per fold: galleries = per-person mean of true-label class contrasts of their training sessions (core1 PC view,
per-session standardised); population contrast = mean of the galleries. Test window t (t >= t_start): probe contrast
from windows 0..t-1 weighted by the B2+TSEM posteriors (soft); score = residual cosine; best gallery p*; weight
w_t = w_max * clip((s* - tau) / ramp, 0, 1); output = log-linear blend of TSEM output and p*'s personal LR head
(PC256, C 0.01, per-session standardised training rows, causal standardisation at test). Nothing else changes."""
import numpy as np
from sklearn.linear_model import LogisticRegression
import rlib as L

FOLDS = {"all_loso": L.PEOPLE, "seen_session": ("P04_s23", "P10_s23")}


def causal_std(F):
    out = np.empty_like(F)
    for t in range(len(F)):
        A = F[:t + 1]; m = A.mean(0)
        out[t] = (F[t] - m) / np.maximum(A.std(0), 1e-3) if t >= 4 else F[t] - m
    return out


def blend(P, Q, w):
    Lg = np.log(np.clip(P, 1e-9, 1)) + w * np.log(np.clip(Q, 1e-9, 1)); Lg -= Lg.max(-1, keepdims=True)
    E = np.exp(Lg); return E / E.sum(-1, keepdims=True)


def fold_model(proto, f, npc=256, Cr=0.01, gal_first_only=False, oof=None, src="q70"):
    """oof: global OOF posteriors (all_loso B2+TSEM) -> pseudo-label bias b (person-equal mean of pseudo - true)."""
    tr, te = L.fold(proto, f); per = {}; itr = np.load(f"{L.FE}/{proto}-{f}.npz")["idx_tr"]; st = L.STREAM[itr]; bias = {}
    for s, (X, y) in tr.items():
        per.setdefault(L.PER(s), []).append((s, X, y))
        if oof is not None:
            Po = oof[itr[st == s]]
            bias.setdefault(L.PER(s), []).append(L.contrast(X, y, w=L.weights(Po, len(y), src, y), npc=npc) - L.contrast(X, y, npc=npc))
    gal, heads = {}, {}
    for p, v in per.items():
        use = [x for x in v if L.SNO(x[0]) == 1] if gal_first_only else v
        gal[p] = np.mean([L.contrast(X, y, npc=npc) for _, X, y in use], 0)
        Xh = np.concatenate([(X[:, :256] - X[:, :256].mean(0)) / np.maximum(X[:, :256].std(0), 1e-3) for _, X, _ in v])
        heads[p] = LogisticRegression(C=Cr, max_iter=3000).fit(Xh, np.concatenate([y for _, _, y in v]))
    pop = np.mean(list(gal.values()), 0)
    b = np.mean([np.mean(v, 0) for v in bias.values()], 0) if bias else 0.0
    return te, gal, heads, pop, b


def scores_session(X, T, gal, pop, npc=256, t_min=10, src="q70", b=0.0):
    """Running residual-cosine scores (n, G): row t uses pseudo-labels (src) of windows 0..t-1 (NaN before t_min)."""
    S = np.full((len(X), len(gal)), np.nan); G = list(gal.values())
    for t in range(t_min, len(X)):
        c = L.contrast(X[:t], None, w=L.weights(T, t, src), npc=npc)
        if c is not None:
            c = c - b
            S[t] = [L.score(c, g, pop, "res") for g in G]
    return S


def route_session(T, S, H, names, tau, ramp, wmax, t_start):
    """-> routed output, per-window weight, per-window routed person (or '')."""
    out = T.copy(); W = np.zeros(len(T)); who = np.array([""] * len(T), dtype=object)
    for t in range(t_start, len(T)):
        j = int(np.nanargmax(S[t])); w = wmax * np.clip((S[t, j] - tau) / ramp, 0, 1)
        if w > 0:
            out[t] = blend(T[t], H[names[j]][t], w); W[t] = w; who[t] = names[j]
    return out, W, who
