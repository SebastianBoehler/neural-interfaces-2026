"""PSYS (ledger s112, explore): a whole personal system for a returning person, built only from that person's own data:
own-space tangent features (causal recentring with the public mean as prior), 64 components fitted on the public
runs, an LDA whose class means adapt inside the session by soft EM from the public class means (pseudo-count M0),
covariance shrunk and fixed from the public runs. Proxy of s82 (runs 1-3 public, 4-6 later). Scored alone, then fused
into the in-population chain against the v5.1 stage. Usage: python studies/12nov/psys.py [smoke]"""
import os, sys
import numpy as np
from sklearn.decomposition import PCA
F = os.path.abspath(__file__ + "/../inpop.py"); exec(compile(open(F).read().split("\nMEM = {}")[0], F, "exec"))  # noqa: S102
SMOKE = len(sys.argv) > 1 and sys.argv[1] == "smoke"; PEOPLE_ = PEOPLE[:2] if SMOKE else PEOPLE; lg = lambda P: np.log(np.clip(P, 1e-3, 1))  # noqa: E731


def personal_system(a, r, M0=20.0, shrink=0.5):
    pub, _ = O.recentred(a); prior = [O.recentred(a)[1][b] for b in range(len(O.BANDS))]
    Fa = O.tang(pub); Fr = O.tang(O.recentred(r, prior)[0]); mu, sd = Fa.mean(0), Fa.std(0) + 1e-6
    pca = PCA(64, random_state=0).fit((Fa - mu) / sd); Xa, Xr = pca.transform((Fa - mu) / sd), pca.transform((Fr - mu) / sd)
    ya = O.y[a]; M = np.stack([Xa[ya == k].mean(0) for k in range(3)]); R = Xa - M[ya]; S = np.cov(R.T); S = (1 - shrink) * S + shrink * np.eye(64) * np.trace(S) / 64
    Si = np.linalg.inv(S); out = np.zeros((len(r), 3)); acc = np.zeros((3, 64)); w = np.zeros(3)
    for t in range(len(r)):
        Mt = (M0 * M + acc) / (M0 + w[:, None]); d = Xr[t] - Mt; L = -0.5 * np.einsum("kd,de,ke->k", d, Si, d); p = np.exp(L - L.max()); p /= p.sum()
        out[t] = p; acc += p[:, None] * Xr[t]; w += p  # causal: window t's own posterior updates the means for t+1
    return out


SYS = {}
for q in PEOPLE_:
    a = np.flatnonzero((O.PER == q) & (O.SES == "1") & (RUN <= 3)); r = B[O.PER[B] == q]
    SYS[q] = (r, personal_system(a, r), O.Personal([a]).stream(r)); print("built", q, f"alone {O.ba(r, SYS[q][1]):.1f}", flush=True)
for bn in ("B2_pool",) if SMOKE else ("B2_pool", "B2_s345", "B2_s012"):
    P0 = chain(bn, True)
    for tag, beta, use in (("v5.1 stage (T.C.S, 0.10)", 0.10, "stage"), ("personal system, 0.25", 0.25, "sys"), ("personal system, 0.50", 0.50, "sys"), ("stage + system 0.25", 0.25, "both")):
        d = []
        for q in PEOPLE_:
            r, Ps, m = SYS[q]; k = O.PER[B] == q; base = np.log(np.clip(P0[k], 1e-9, 1))
            L = base + (0.10 * sum(lg(m[x]) for x in "TCS") / 3 if use in ("stage", "both") else 0) + (beta * lg(Ps) if use in ("sys", "both") else 0)
            e = np.exp(L - L.max(1, keepdims=True)); d.append(O.ba(r, e / e.sum(1, keepdims=True)) - O.ba(r, P0[k]))
        d = np.array(d); print(f"{bn} {tag:26s}: {d.mean():+.2f} | " + " ".join(f"{x:+.1f}" for x in d), flush=True)
print("alone: personal system " + " ".join(f"{q[1:]}:{O.ba(SYS[q][0], SYS[q][1]):.0f}" for q in PEOPLE_))
