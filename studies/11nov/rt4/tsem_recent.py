"""TSEM-R (ledger s23): TSEM seeds restricted to the last N windows; shipped chain (routing v3), exploration bases."""
import sys
import numpy as np
REPO = __import__("os").path.abspath(__file__ + "/../../../..")
sys.path.insert(0, REPO + "/replay")
import env
A = env.A; SC = A.SC; G = A.L.G; T_ = G.T; lik0 = G.tsem_lik


def lik_recent(N):
    def f(P, sess, X, rho=0.7, shrink=0.7, m0=20, nmin=15):
        Lk = np.full_like(P, np.nan); tt = np.zeros(len(P))
        for s in np.unique(sess):
            r = np.flatnonzero(sess == s); lab = P[r].argmax(1); conf = P[r].max(1)
            for t in range(nmin, len(r)):
                A_, fx = X[r[:t]], X[r[t]]; m, sd = A_.mean(0), np.maximum(A_.std(0), 1e-3)
                Zm, z = (A_ - m) / sd, (fx - m) / sd; lo = max(0, t - N); n = t - lo
                q = max(2, int(rho * n / 3)); sel = []
                for k in range(3):
                    ik = lo + np.flatnonzero(lab[lo:t] == k); sel.append(ik[np.argsort(-conf[ik])[:q]])
                if min(len(x) for x in sel) < 2:
                    continue
                mu = np.stack([Zm[x].sum(0) / (len(x) + m0) for x in sel])
                R = np.concatenate([Zm[x] - mu[k] for k, x in enumerate(sel)])
                Cv = np.cov(R.T); Cv = (1 - shrink) * Cv + shrink * np.eye(len(Cv)) * np.trace(Cv) / len(Cv)
                Lk[r[t]] = T_._lda(z[None], mu, np.linalg.inv(Cv))[0]; tt[r[t]] = t
        return Lk, tt
    return f


POS = {}
for p in A.PROTOS:
    ss = SC.B[f"sess|{p}"]; pos = np.zeros(len(ss), int)
    for s in np.unique(ss):
        m = np.flatnonzero(ss == s); pos[m] = np.arange(len(m))
    POS[p] = pos
bases = sys.argv[1].split(",") if len(sys.argv) > 1 else list(env.EXPL); res = {}
for N in (10 ** 6, 120, 90, 60):
    G.tsem_lik = lik0 if N == 10 ** 6 else lik_recent(N); dn, ds, late = [], [], []
    for bn in bases:
        r = A.system(bn, A.FAMS); s = A.summary(r); dn.append(s["dn"]); ds.append(s["ds"])
        y = SC.B["y|all_loso"]; ok = r["out"]["all_loso"].argmax(1) == y; late.append(ok[POS["all_loso"] >= 90].mean())
    res[N] = (np.mean(dn, 0), np.mean(ds, 0), np.mean(late)); d0 = res[10 ** 6]
    print(f"N {'all' if N == 10 ** 6 else N}: J {0.5 * res[N][0].mean() + 0.5 * res[N][1].mean():.4f} dJ {0.5 * (res[N][0] - d0[0]).mean() + 0.5 * (res[N][1] - d0[1]).mean():+.2f} "
          f"new {(res[N][0] - d0[0]).mean():+.2f} ({int((res[N][0] >= d0[0]).sum())}/10) seen {(res[N][1] - d0[1]).mean():+.2f} | new-people accuracy windows >= 90: {res[N][2]:.4f}", flush=True)
G.tsem_lik = lik0
