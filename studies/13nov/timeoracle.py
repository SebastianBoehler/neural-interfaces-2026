"""TIMEORACLE (ledger s170): SessionEM and TSEM with every delivered window weighted by exp(-|recording-position
distance to the current window| / H): the templates (means, scales, covariances) use the whole delivered set, soft,
causal over the delivery stream. ORACLE: the recording position is not available at evaluation time; form A would put a
slow-state EEG similarity in its place. H = inf reproduces the shipped stages exactly (asserted by the drivers).
install(tau, H) patches gm.em_g and grid.tsem_lik (both tails and the shadow call them at run time); uninstall() restores.
tau = {n_rows: recording position of each current row} (the stores' rows are in delivery order)."""
import numpy as np
import gm
import grid as G
import mech as M
import session_em as SE
_EM0, _TL0 = gm.em_g, G.tsem_lik
ST = {"tau": None, "H": None}


def _w(taus, tc):
    w = np.exp(-np.abs(taus - tc) / ST["H"]); return w * len(w) / w.sum()  # mean weight 1: effective counts stay comparable


def _wstats(A, w):
    m = (w[:, None] * A).sum(0) / w.sum(); sd = np.sqrt(np.maximum((w[:, None] * (A - m) ** 2).sum(0) / w.sum(), 0)); return m, sd


class EMw(gm.EMg):
    def _adapt(self, A, B, f, p, t):
        w = _w(np.asarray(self.T[:len(A)], float), self.tc)
        m, sd = _wstats(A, w); sd = np.maximum(sd, SE.SD_FLOOR); Z, z = (A - m) / sd, (f - m) / sd; Q = B
        for _ in range(self.iters):
            Qw = Q * w[:, None]; mu = (Qw / (Qw.sum(0, keepdims=True) + self.m0)).T @ Z
            C = np.cov((Z - Q @ mu).T, aweights=w); C = (1 - self.shrink) * C + self.shrink * np.eye(len(C)) * np.trace(C) / len(C)
            Ci = np.linalg.inv(C); Q = B * SE._softmax_lda(Z, mu, Ci) ** self.lam; Q = Q / Q.sum(1, keepdims=True)
        e = self.lam * t / (t + self.n0) * (1 + self.bem * (M.h(t, 30) * self.g))
        q = p * SE._softmax_lda(z[None], mu, Ci)[0] ** e; return q / q.sum()


def em_g(b, proto, g, bem=gm.BEM, nmin=gm.NMIN):
    F, sess = gm.L.Z[f"GE|{proto}"], gm.L.Z[f"sess|{proto}"]; tau = ST["tau"][len(b)]; out = np.empty_like(b)
    for s in dict.fromkeys(sess):
        e = EMw(bem, lam=0.5, n0=60, m0=20, shrink=0.7, nmin=nmin); e.T = []
        for r in np.flatnonzero(sess == s):
            e.g = g[r]; e.tc = tau[r]; out[r] = e.step(F[r], b[r]); e.T.append(tau[r])
    return out


def tsem_lik(P, sess, X, rho=0.7, shrink=0.7, m0=20, nmin=15):
    tau = ST["tau"][len(P)]; L = np.full_like(P, np.nan); tt = np.zeros(len(P))
    for s in np.unique(sess):
        r = np.flatnonzero(sess == s); lab = P[r].argmax(1); conf = P[r].max(1)
        for t in range(nmin, len(r)):
            A_, f = X[r[:t]], X[r[t]]; w = _w(tau[r[:t]].astype(float), tau[r[t]])
            m, sd = _wstats(A_, w); sd = np.maximum(sd, 1e-3); Zm, z = (A_ - m) / sd, (f - m) / sd
            q = max(2, int(rho * t / 3)); sel = []
            for k in range(3):
                ik = np.flatnonzero(lab[:t] == k); sel.append(ik[np.argsort(-conf[ik])[:q]])
            if min(len(x) for x in sel) < 2:
                continue
            mu = np.stack([(w[x, None] * Zm[x]).sum(0) / (w[x].sum() + m0) for x in sel])
            R = np.concatenate([Zm[x] - mu[k] for k, x in enumerate(sel)]); wr = np.concatenate([w[x] for x in sel])
            Cv = np.cov(R.T, aweights=wr); Cv = (1 - shrink) * Cv + shrink * np.eye(len(Cv)) * np.trace(Cv) / len(Cv)
            L[r[t]] = G.T._lda(z[None], mu, np.linalg.inv(Cv))[0]; tt[r[t]] = t
    return L, tt


def install(tau, H):
    ST["tau"], ST["H"] = tau, float(H); gm.em_g = em_g; G.tsem_lik = tsem_lik


def uninstall():
    gm.em_g, G.tsem_lik = _EM0, _TL0; ST["tau"] = ST["H"] = None
