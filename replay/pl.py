"""Personal-model team (gu2/personal, 7 Nov): shared library.
- pipe(): B2 tail from the pre-rebalancing mix (rebal a1.15 W30 -> gaze/event SessionEM l0.5 -> TSEM pc128 l0.15),
  optionally with personal template priors in SessionEM / TSEM (prior class means, pseudo-count m0p).
- solve(): exact linear personal head in the span of the own training rows (kernel form) with offset logits f0
  (MAP pull to a population head when f0 = its logits; personal-only head when f0 = 0).
- scoring: person-equal cell-mean BA (tsem_confirm/score.py definition), per person and per session."""
import os
import numpy as np
from scipy.optimize import minimize
import paths
S, REPO = paths.DATA, paths.REPO
import tsem as TS
import session_em as SE
from ensemble import rebalance

B = dict(np.load(S + "/bases.npz", allow_pickle=True))
BF = dict(np.load(S + "/bases_fresh.npz", allow_pickle=True))
KD = os.environ.get("KD", S + "/kern")
with np.load(S + "/base_cache.npz") as f:
    GE_ALL = np.c_[f["G"], f["EV"]]
SEEN = ["P04", "P10"]; PEOPLE = [f"P{i:02d}" for i in range(1, 11)]
_PC = {}


def pc(proto, npc=128):
    if (proto, npc) not in _PC:
        _PC[(proto, npc)] = TS.load_feats(S + "/feats_exact", B[f"idx|{proto}"], B[f"sess|{proto}"],
                                          npc=npc, proto=proto)[0]
    return _PC[(proto, npc)]


BM = dict(np.load(S + "/bases_fresh_mix.npz")) if os.path.exists(S + "/bases_fresh_mix.npz") else {}


def get(bn, key, proto):
    k = f"{bn}|{key}|{proto}"
    return B[k] if k in B else BM[k] if k in BM else BF[k]


def norm(L):
    L = L - L.max(1, keepdims=True); E = np.exp(L); return E / E.sum(1, keepdims=True)


def blend(P, Q, w):
    w = np.broadcast_to(np.asarray(w, float), (len(P),))[:, None]
    return norm(np.log(np.clip(P, 1e-9, 1)) + w * np.log(np.clip(Q, 1e-9, 1)))


def slot_a(bn, proto):
    """mix = a * slot + (1 - a) * deep -> a (slot weight share)."""
    m, s, dp = (get(bn, k, proto) for k in ("mix", "slot", "deep"))
    a = ((m - dp) * (s - dp)).sum() / ((s - dp) ** 2).sum()
    assert np.abs(a * s + (1 - a) * dp - m).max() < 1e-9, "mix is not a*slot+(1-a)*deep"
    return a


def rebal(P, proto):
    st = B[f"stream|{proto}"]; out = np.empty_like(P)
    for s in dict.fromkeys(st):
        r = st == s; out[r] = rebalance(P[r], 1.15, 5.0, 30)
    return out


def se_prior(F, P, streams, priors=None, m0p=0.0, lam=0.5, n0=60, m0=20, shrink=0.7, iters=2, nmin=10):
    """SessionEM.apply with optional prior class means {stream: (3, d) in the session-standardised view}."""
    if not priors or m0p == 0:
        return SE.apply(F, P, streams, lam=lam, n0=n0, m0=m0, shrink=shrink)
    out = np.empty_like(P)
    for s in dict.fromkeys(streams):
        r = np.flatnonzero(streams == s); mp = priors.get(s); w0 = m0p if mp is not None else 0.0
        mp = np.zeros((3, F.shape[1])) if mp is None else mp
        for t, i in enumerate(r):
            out[i] = P[i]
            if t < nmin:
                continue
            A, Bp, f, p = F[r[:t]], P[r[:t]], F[i], P[i]
            m, sd = A.mean(0), np.maximum(A.std(0), SE.SD_FLOOR); Z, z = (A - m) / sd, (f - m) / sd
            Q = Bp
            for _ in range(iters):
                mu = (Q.T @ Z + w0 * mp) / (Q.sum(0)[:, None] + m0 + w0)
                Cv = np.cov((Z - Q @ mu).T); Cv = (1 - shrink) * Cv + shrink * np.eye(len(Cv)) * np.trace(Cv) / len(Cv)
                try:
                    Ci = np.linalg.inv(Cv)
                except np.linalg.LinAlgError:
                    Ci = None; break
                Q = Bp * SE._softmax_lda(Z, mu, Ci) ** lam; Q = Q / Q.sum(1, keepdims=True)
            if Ci is None:
                continue
            e = lam * t / (t + n0); q = p * SE._softmax_lda(z[None], mu, Ci)[0] ** e
            out[i] = q / q.sum() if np.isfinite(q).all() else p
    return out


def tsem(P, proto, priors=None, m0p=0.0, lam=0.15, rho=0.7, shrink=0.7, m0=20, n0=60, nmin=15):
    """TSEM on the exact pc128 view; priors {sess id: (3, 128) standardised class means} pull the class means."""
    if not priors or m0p == 0:
        return TS.tsem(P, B[f"sess|{proto}"], TS.static(pc(proto)), lam=lam)
    sess, V, out = B[f"sess|{proto}"], pc(proto), P.copy()
    for s in np.unique(sess):
        r = np.flatnonzero(sess == s); lab = P[r].argmax(1); conf = P[r].max(1); mp = priors.get(s)
        for t in range(nmin, len(r)):
            A_, f = V[r[:t]], V[r[t]]; m, sd = A_.mean(0), np.maximum(A_.std(0), 1e-3); Z, z = (A_ - m) / sd, (f - m) / sd
            q = max(2, int(rho * t / 3)); sel = []
            for k in range(3):
                ik = np.flatnonzero(lab[:t] == k); sel.append(ik[np.argsort(-conf[ik])[:q]])
            if min(len(x) for x in sel) < 2 and mp is None:
                continue
            pm = mp if mp is not None else np.zeros((3, V.shape[1])); w0 = m0p if mp is not None else 0.0
            mu = np.stack([(Z[x].sum(0) + w0 * pm[k]) / (len(x) + m0 + w0) for k, x in enumerate(sel)])
            R = np.concatenate([Z[x] - mu[k] for k, x in enumerate(sel)])
            Cv = np.cov(R.T) if len(R) > 3 else np.eye(V.shape[1])
            Cv = (1 - shrink) * Cv + shrink * np.eye(len(Cv)) * np.trace(Cv) / len(Cv)
            e = lam * t / (t + n0); qq = P[r[t]] * TS._lda(z[None], mu, np.linalg.inv(Cv))[0] ** e
            out[r[t]] = qq / qq.sum()
    return out


def pipe(mix, proto, se_pri=None, se_m0p=0.0, ts_pri=None, ts_m0p=0.0, ts=True):
    e = se_prior(B[f"GE|{proto}"], rebal(mix, proto), B[f"stream|{proto}"], se_pri, se_m0p)
    return tsem(e, proto, ts_pri, ts_m0p) if ts else e


def solve(K, f0, y, tr, C, sw=None, out=None):
    """min_W sum_i sw_i CE(f0_i + W z_i + b) + |W|^2 / (2C) over the training rows tr (indices into K) -> logits
    of rows `out` (default all). Exact: W lies in the span of the training rows (eigen-basis of K[tr, tr])."""
    lam, U = np.linalg.eigh(K[np.ix_(tr, tr)]); keep = lam > lam.max() * 1e-10
    out = np.arange(len(K)) if out is None else out
    M = U[:, keep] / np.sqrt(lam[keep]); X = K[np.ix_(tr, tr)] @ M; Xo = K[np.ix_(out, tr)] @ M
    r = X.shape[1]; Y1 = np.eye(3)[y]; w = np.ones(len(tr)) if sw is None else sw; F0 = f0[tr]

    def fg(th):
        Bm, b = th[:3 * r].reshape(r, 3), th[3 * r:]
        L = F0 + X @ Bm + b; L = L - L.max(1, keepdims=True); lp = L - np.log(np.exp(L).sum(1, keepdims=True))
        P = np.exp(lp); G = w[:, None] * (P - Y1)
        loss = -(w * (lp * Y1).sum(1)).sum() + (Bm ** 2).sum() / (2 * C) + 1e-6 * (b ** 2).sum()
        return loss, np.concatenate([(X.T @ G + Bm / C).ravel(), G.sum(0) + 2e-6 * b])
    th = minimize(fg, np.zeros(3 * r + 3), jac=True, method="L-BFGS-B", options=dict(maxiter=5000)).x
    return f0[out] + Xo @ th[:3 * r].reshape(r, 3) + th[3 * r:]


def kern(core, tag):
    z = dict(np.load(f"{KD}/{core}-{tag}.npz"))
    off = np.concatenate([[0], np.cumsum(z["n"])]); z["sl"] = {g: np.arange(off[i], off[i + 1]) for i, g in enumerate(z["groups"])}
    return z


def cat(z, key, groups):
    return np.concatenate([z[f"{key}|{g}"] for g in groups])


def cell_ba(y, pred):
    return 100 * np.mean([np.mean(pred[y == k] == k) for k in np.unique(y)])


def scores(P, proto, rows=None):
    """-> ({person: BA}, {person s<k>: BA}) over cells restricted to `rows` (bool mask over base rows)."""
    y, cell, per, ss = B[f"y|{proto}"], B[f"cell|{proto}"], B[f"person|{proto}"], B[f"sess|{proto}"]
    m0 = np.ones(len(y), bool) if rows is None else rows; pred = P.argmax(1); pp, sp = {}, {}
    for c in np.unique(cell[m0]):
        m = m0 & (cell == c); p = per[m][0]; v = cell_ba(y[m], pred[m])
        pp.setdefault(p, []).append(v); sp.setdefault((p, int(ss[m][0])), []).append(v)
    return {k: np.mean(v) for k, v in pp.items()}, {k: np.mean(v) for k, v in sp.items()}


def delta(P, ref, proto="seen_session", rows=None):
    a, sa = scores(P, proto, rows); b, sb = scores(ref, proto, rows)
    return {k: a[k] - b[k] for k in a}, {k: sa[k] - sb[k] for k in sa}


def pos(proto):
    return {int(i): j for j, i in enumerate(B[f"idx|{proto}"])}
