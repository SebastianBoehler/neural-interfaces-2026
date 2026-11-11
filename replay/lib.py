"""Early-window study: shared helpers. Chain = mix(ws(t)) -> rebal -> SessionEM -> TSEM (replica of retune/grid.py)."""
import os, sys
import paths
import numpy as np
import grid as G  # loads retune/inputs.npz as G.Z
import session_em as SE
import score as SC

Z = G.Z; PROTOS = G.PROTOS; PEOPLE, SEEN, G1, G2 = SC.PEOPLE, SC.SEEN, SC.G1, SC.G2
_D = np.load(paths.WINDOWS)
RUN, IFACE = _D["run"], _D["interface"]
BINS = [(0, 10), (10, 20), (20, 40), (40, 60), (60, 120), (120, 10**6)]


def pos(proto):
    s = Z[f"sess|{proto}"]; out = np.empty(len(s), int)
    for v in np.unique(s):
        r = np.flatnonzero(s == v); assert (np.diff(r) > 0).all(); out[r] = np.arange(len(r))
    return out


def runpos(proto):
    """position within run (run = contiguous block of same run/interface within session) — DIAGNOSTIC ONLY (metadata)."""
    idx, s = Z[f"idx|{proto}"], Z[f"sess|{proto}"]; key = np.char.add(RUN[idx].astype(str), IFACE[idx])
    out = np.zeros(len(s), int)
    for i in range(1, len(s)):
        out[i] = out[i - 1] + 1 if (s[i] == s[i - 1] and key[i] == key[i - 1]) else 0
    return out


def mixw(Ps, W):
    """W: (n_members,) or (n_rows, n_members) weights -> weighted mean of member posteriors."""
    if W.ndim == 1:
        W = np.broadcast_to(W, (Ps.shape[1], len(W)))
    W = W / W.sum(1, keepdims=True)
    return np.einsum("rm,mrc->rc", W, Ps)


def rebal(P, sess, alpha=1.15, k=5.0, W=30, prior=None, kfun=None, afun=None):
    """diag_common.rebal generalised: prior class shares (default uniform), k / alpha may depend on position n."""
    prior = np.full(3, 1 / 3) if prior is None else np.asarray(prior)
    out = np.empty_like(P)
    for s in np.unique(sess):
        r = np.flatnonzero(sess == s); X = P[r]
        cs = np.vstack([np.zeros((1, 3)), np.cumsum(X, 0)])
        n = np.arange(len(X)); lo = np.maximum(0, n - W)
        kk = (np.full(len(X), k) if kfun is None else kfun(n))[:, None]
        aa = (np.full(len(X), alpha) if afun is None else afun(n))[:, None]
        m = (kk * prior + cs[n] - cs[lo]) / (kk[:, 0] + (n - lo))[:, None]
        q = X / m ** aa
        out[r] = q / q.sum(1, keepdims=True)
    return out


def em(P, proto, lam=0.5, n0=60, nmin=10):
    return SE.apply(Z[f"GE|{proto}"], P, Z[f"sess|{proto}"], lam=lam, n0=n0, m0=20, shrink=0.7, nmin=nmin)


def tsem(P, proto, lam=0.15, n0=60):
    L, tt = G.tsem_lik(P, Z[f"sess|{proto}"], Z[f"pc128|{proto}"], rho=0.7)
    return G.apply_lam(P, L, tt, lam, n0=n0)


def chain(base, proto, W=None, rb=None, emkw=None, tskw=None, stages=False):
    Ps = Z[f"{base}|Ps|{proto}"]; W = Z[f"{base}|ws|{proto}"] if W is None else W
    mix = mixw(Ps, W); b = rebal(mix, Z[f"sess|{proto}"], **(rb or {})); e = em(b, proto, **(emkw or {}))
    t = tsem(e, proto, **(tskw or {}))
    return (mix, b, e, t) if stages else t


def person(P, proto):
    return SC.person(P, proto)


def J(Pn, Ps_):
    return 0.5 * person(Pn, "all_loso").mean() + 0.5 * person(Ps_, "seen_session").mean()
