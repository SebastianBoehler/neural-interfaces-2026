"""TSEM (confident-seed tangent-space session EM), shared by the confirmation scripts (6 Nov).
em_conf(): verbatim copy of gu/self_training/run2_emconf.em_conf (static view; seeds = the stage's input posteriors).
tsem(): same estimator with (i) a per-window view callback (for views that change with t, e.g. causal session PCA),
(ii) an optional separate seed-ranking score, (iii) optional per-column scaling after standardisation.
For a static view, rank=None and colscale=None it is identical to em_conf (checked in confirm.py)."""
import glob
import numpy as np


def _lda(Z, mu, Ci):
    W = Ci @ mu.T
    s = Z @ W - 0.5 * (mu * W.T).sum(1)
    s -= s.max(-1, keepdims=True); g = np.exp(s)
    return g / g.sum(-1, keepdims=True)


def em_conf(Fv, P, seeds, sess, lam, rho, shrink=0.7, m0=5, n0=60, nmin=15, recursive=False):
    out = P.copy()
    for s in np.unique(sess):
        r = np.flatnonzero(sess == s); conf = seeds[r].max(1); lab = seeds[r].argmax(1)
        for t in range(nmin, len(r)):
            if recursive:
                conf = out[r].max(1); lab = out[r].argmax(1)
            A_, f = Fv[r[:t]], Fv[r[t]]
            m, sd = A_.mean(0), np.maximum(A_.std(0), 1e-3)
            Z, z = (A_ - m) / sd, (f - m) / sd
            q = max(2, int(rho * t / 3)); sel = []
            for k in range(3):
                ik = np.flatnonzero(lab[:t] == k); sel.append(ik[np.argsort(-conf[ik])[:q]])
            if min(len(x) for x in sel) < 2:
                continue
            mu = np.stack([Z[x].sum(0) / (len(x) + m0) for x in sel])
            R = np.concatenate([Z[x] - mu[k] for k, x in enumerate(sel)])
            Cv = np.cov(R.T); Cv = (1 - shrink) * Cv + shrink * np.eye(len(Cv)) * np.trace(Cv) / len(Cv)
            e = lam * t / (t + n0) if n0 else lam
            qq = P[r[t]] * _lda(z[None], mu, np.linalg.inv(Cv))[0] ** e
            out[r[t]] = qq / qq.sum()
    return out


def tsem(P, sess, view, lam=0.15, rho=0.7, shrink=0.7, m0=20, n0=60, nmin=15, rank=None, colscale=None):
    """view(s, r, t) -> (A_ (t, d) views of past windows r[:t], f (d,) view of window r[t])."""
    out = P.copy()
    if lam == 0:
        return out
    for s in np.unique(sess):
        r = np.flatnonzero(sess == s); lab = P[r].argmax(1)
        conf = P[r].max(1) if rank is None else rank[r]
        for t in range(nmin, len(r)):
            A_, f = view(s, r, t)
            m, sd = A_.mean(0), np.maximum(A_.std(0), 1e-3)
            Z, z = (A_ - m) / sd, (f - m) / sd
            if colscale is not None:
                Z, z = Z * colscale, z * colscale
            q = max(2, int(rho * t / 3)); sel = []
            for k in range(3):
                ik = np.flatnonzero(lab[:t] == k); sel.append(ik[np.argsort(-conf[ik])[:q]])
            if min(len(x) for x in sel) < 2:
                continue
            mu = np.stack([Z[x].sum(0) / (len(x) + m0) for x in sel])
            R = np.concatenate([Z[x] - mu[k] for k, x in enumerate(sel)])
            Cv = np.cov(R.T); Cv = (1 - shrink) * Cv + shrink * np.eye(len(Cv)) * np.trace(Cv) / len(Cv)
            e = lam * t / (t + n0) if n0 else lam
            qq = P[r[t]] * _lda(z[None], mu, np.linalg.inv(Cv))[0] ** e
            out[r[t]] = qq / qq.sum()
    return out


def static(Fv):
    return lambda s, r, t: (Fv[r[:t]], Fv[r[t]])


def load_feats(directory, idx, sess, npc=256, proto=None):
    """-> pc (n, npc) aligned to idx rows, K {session id: Gram (n_s, n_s)} in row order."""
    pos = {int(g): i for i, g in enumerate(idx)}; pc = np.full((len(idx), npc), np.nan); K = {}
    assert proto in ("all_loso", "seen_session")
    for f in sorted(glob.glob(f"{directory}/{proto}-*.npz")):
        z = np.load(f)
        for k in range(len(z["sessions"])):
            r = np.array([pos[int(i)] for i in z[f"{k}|idx"]])
            assert (np.diff(r) == 1).all()
            pc[r] = z[f"{k}|pc"][:, :npc]; s = int(sess[r[0]]); assert (sess[r] == s).all(); K[s] = z[f"{k}|K"]
    assert not np.isnan(pc).any()
    return pc, K
