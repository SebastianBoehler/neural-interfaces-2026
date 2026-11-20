"""Lane G1: class-coupled SessionEM. anchors(): source class anchors in prefix-standardized GE space by prefix age band; coupled_cls(): EMg subclass with the graph update.
mu = A + D, (N + c I + ew L) D = S - N A (edge arms), c = shipped m0. Modes: zero (A=0, L=0: bitwise base), anchor (L=0), shared (one D), edge (Laplacian of one edge)."""
import numpy as np

EDGES = (40, 80)  # band of prefix size t: <40 (20-39), 40-79, 80+
SD_FLOOR = 1e-3


def band(t):
    return int(t >= EDGES[0]) + int(t >= EDGES[1])


def session_band_means(F, y, tmin=20):
    """one session in delivery order -> (3 bands, 3 classes, d) mean over prefix sizes t (strictly past rows) of the class mean z; nan where a class is absent."""
    n, d = F.shape
    cs, cq = np.vstack([np.zeros(d), np.cumsum(F, 0)]), np.vstack([np.zeros(d), np.cumsum(F * F, 0)])
    oh = np.eye(3)[y]
    ck = np.concatenate([np.zeros((1, 3, d)), np.cumsum(oh[:, :, None] * F[:, None, :], 0)])
    cn = np.vstack([np.zeros(3), np.cumsum(oh, 0)])
    out, cnt = np.zeros((3, 3, d)), np.zeros((3, 3))
    for t in range(tmin, n + 1):
        m = cs[t] / t
        sd = np.maximum(np.sqrt(np.maximum(cq[t] / t - m * m, 0.0)), SD_FLOOR)
        ok = cn[t] >= 1
        z = (ck[t] / np.maximum(cn[t], 1)[:, None] - m) / sd
        b = band(t)
        out[b][ok] += z[ok]; cnt[b][ok] += 1
    with np.errstate(invalid="ignore", divide="ignore"):
        return out / cnt[:, :, None]


def anchors(F, y, sess, person, held):
    """A (3, 3, d) from the labelled rows of all people except `held`: per-person mean of per-session means, equal person weight, per band and class."""
    per = []
    for p in dict.fromkeys(person):
        if p == held:
            continue
        ms = [session_band_means(F[r], y[r]) for s in dict.fromkeys(sess[person == p]) for r in [np.flatnonzero((sess == s) & (person == p))]]
        with np.errstate(invalid="ignore"):
            per.append(np.nanmean(np.stack(ms), 0))
    A = np.nanmean(np.stack(per), 0)
    assert np.isfinite(A).all(), "anchor band/class with no source rows"
    return A


def lap(edge):
    L = np.zeros((3, 3))
    if edge:
        a, b = edge
        L[a, a] = L[b, b] = 1.0; L[a, b] = L[b, a] = -1.0
    return L


def coupled_cls(base):
    class EMc(base):
        """EMg whose class means come from the coupled update; cfg = dict(mode, edge, ew, A (3,3,d) or None)."""
        cfg = None

        def _adapt(self, A_, B, f, p, t):
            SE, h = self.mods
            m, sd = A_.mean(0), np.maximum(A_.std(0), SE.SD_FLOOR)
            Z, z = (A_ - m) / sd, (f - m) / sd
            Q = B
            cfg = self.cfg
            Anc = np.zeros((3, Z.shape[1])) if cfg["mode"] == "zero" else cfg["A"][band(t)]
            for _ in range(self.iters):
                if cfg["mode"] in ("zero", "anchor"):
                    den = Q.sum(0, keepdims=True) + self.m0
                    mu = (Q / den).T @ Z + (self.m0 / den[0])[:, None] * Anc
                else:
                    N, S = Q.sum(0), Q.T @ Z
                    R = S - N[:, None] * Anc
                    if cfg["mode"] == "shared":
                        mu = Anc + (R.sum(0) / (N.sum() + 3 * self.m0))[None]
                    else:
                        mu = Anc + np.linalg.solve(np.diag(N + self.m0) + cfg["ew"] * lap(cfg["edge"]), R)
                C = np.cov((Z - Q @ mu).T)
                C = (1 - self.shrink) * C + self.shrink * np.eye(len(C)) * np.trace(C) / len(C)
                Ci = np.linalg.inv(C)
                Q = B * SE._softmax_lda(Z, mu, Ci) ** self.lam
                Q = Q / Q.sum(1, keepdims=True)
            e = self.lam * t / (t + self.n0) * (1 + self.bem * (h(t, 30) * self.g))
            q = p * SE._softmax_lda(z[None], mu, Ci)[0] ** e
            return q / q.sum()
    return EMc
