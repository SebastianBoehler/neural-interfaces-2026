"""Lane G personal-model fits in the package's own array format (one person -> one row of route_ps1_* and own_*).

stock: the recipe of replay/rlib3.ps1_heads + studies/12nov/ownspace.Personal (recorded order, whole-session statistics).
g1: every training visit as R replicas in random delivery order, causal statistics as at test time, replicas 1.. augmented.
Both return {key: array (no person axis)} with exactly the package keys and shapes.
"""
import numpy as np
from scipy.linalg import eigh
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from common import soft, strong, views

IU = np.triu_indices(41)
WT = np.where(IU[0] == IU[1], 1.0, np.sqrt(2.0))


def isqrt(M):
    w, V = np.linalg.eigh(M)
    return np.einsum("...ij,...j,...kj->...ik", V, 1 / np.sqrt(w), V)


def logm(C):
    w, V = np.linalg.eigh(C)
    return np.einsum("...ij,...j,...kj->...ik", V, np.log(w), V)


def tang(Cw):
    """Cw (n, bands, C, C) -> (n, bands * 861), band-major as OwnStage."""
    return np.concatenate([logm(Cw[:, b])[:, IU[0], IU[1]] * WT for b in range(Cw.shape[1])], 1)


def recentre(C, prior, kref, causal):
    """(n, bands, C, C) -> whitened. causal: running mean with the prior as kref windows (OwnStage); else whole session."""
    n = len(C)
    if causal:
        M = (kref * prior[None] + np.cumsum(C, 0)) / (kref + np.arange(1, n + 1))[:, None, None, None]
    else:
        M = np.repeat(C.mean(0, keepdims=True), n, 0)
    W = isqrt(M)
    return W @ C @ W


def stdz(U):
    return (U - U.mean(0)) / np.maximum(U.std(0), 1e-6)


def causal_stdz(U):
    """Route.step's standardisation of the view: running mean/std over the session so far (including this window)."""
    out = np.empty_like(U)
    for t in range(len(U)):
        A = U[:t + 1]
        m = A.mean(0)
        out[t] = (U[t] - m) / np.maximum(A.std(0), 1e-3) if t >= 4 else U[t] - m
    return out


def causal_z(F):
    n = np.arange(1, len(F) + 1)[:, None]
    m = np.cumsum(F, 0) / n
    v = np.cumsum(F ** 2, 0) / n - m ** 2
    return np.where(n >= 4, (F - m) / np.sqrt(np.maximum(v, 1e-9)), 0.0)


def csp(Cw, y):
    """Per band 18 filters from the three class pairs (ownspace.Personal)."""
    Ws = []
    for b in range(Cw.shape[1]):
        M = [Cw[y == k, b].mean(0) for k in range(3)]
        Ws.append(np.concatenate([eigh(M[a], M[a] + M[c])[1][:, [0, 1, 2, -3, -2, -1]] for a, c in ((0, 1), (0, 2), (1, 2))], 1))
    return np.stack(Ws)


def csp_feat(W, Cw):
    return np.concatenate([np.log(np.einsum("cf,ncd,df->nf", W[b], Cw[:, b], W[b])) for b in range(len(W))], 1)


def arrays(streams, kref, causal, view_causal):
    """streams: list of dict(u (n,256), C (n,bands,C,C), sl (n,K), y (n,), clean (n,) bool for the prior's session means).
    -> package row dict. ``causal`` picks the statistics of the own members; ``view_causal`` those of the head."""
    clean = [s for s in streams if s["is_clean"]]
    prior = np.mean([s["C"].mean(0) for s in clean], 0)
    y = np.concatenate([s["y"] for s in streams])
    Cw = np.concatenate([recentre(s["C"], prior, kref, causal) for s in streams])
    head_x = np.concatenate([causal_stdz(s["u"]) if view_causal else stdz(s["u"]) for s in streams])
    head = LogisticRegression(C=0.1, max_iter=3000).fit(head_x, y)
    T = make_pipeline(StandardScaler(), LogisticRegression(C=0.01, max_iter=500)).fit(tang(Cw), y)
    W = csp(Cw, y)
    Cl = LDA(solver="lsqr", shrinkage=0.5).fit(csp_feat(W, Cw), y)
    S = np.concatenate([causal_z(s["sl"]) if causal else (s["sl"] - s["sl"].mean(0)) / (s["sl"].std(0) + 1e-9) for s in streams])
    Sl = LDA(solver="lsqr", shrinkage=0.7).fit(S, y)
    sc, lr = T.steps[0][1], T.steps[1][1]
    return dict(route_ps1_W=head.coef_, route_ps1_c=head.intercept_, own_prior=prior, own_csp=W, own_t_mu=sc.mean_,
                own_t_sd=sc.scale_, own_t_W=lr.coef_, own_t_c=lr.intercept_, own_c_W=Cl.coef_, own_c_c=Cl.intercept_,
                own_s_W=Sl.coef_, own_s_c=Sl.intercept_)


MODES = {  # mode: (shuffled order, augmentation of replicas 1.., replicas, causal statistics)
    "stock": (False, None, 1, False), "g1a": (False, "strong", None, False), "g1c": (True, None, 1, True), "g1": (True, "strong", None, True),
    "g2": (True, "soft", None, True), "g4": (True, "noise", None, True)}
AUG = {"strong": strong, "soft": soft, "noise": lambda X, rng: soft(X, rng, gain=0.0)}


def build_streams(mod, z, feat, D, rows_by_visit, mode, reps, seed):
    """Training streams of one person: each visit ``n`` times; replicas 1.. augmented (the package's `strong`) if the mode
    says so; replica order random if the mode says so, else recorded order."""
    shuffle, aug, n, _ = MODES[mode]
    n = n or reps
    rng = np.random.default_rng(seed)
    out = []
    for visit, rows in rows_by_visit.items():
        for r in range(n):
            order = rng.permutation(rows) if shuffle else np.sort(rows)
            X = D["X"][order].astype(np.float32)
            if aug and r > 0:
                X = AUG[aug](X, rng)
            C, sl = feat(X)
            out.append(dict(u=views(mod, z, X), C=C, sl=sl, y=D["y"][order], is_clean=not (aug and r > 0), visit=visit))
    return out


def fit(mod, z, feat, D, person, visits, mode, reps=5, seed=20261119):
    rows = {v: np.flatnonzero((D["person"] == person) & (D["visit"] == v)) for v in visits}
    causal = MODES[mode][3]
    return arrays(build_streams(mod, z, feat, D, rows, mode, reps, seed), float(z["own_kref"]), causal, causal)
