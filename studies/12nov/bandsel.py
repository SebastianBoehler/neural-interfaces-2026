"""BANDSEL (ledger s116, explore): people carry their classes in different bands (s77). Per-band population models
(tangent T and CSP C, nine bands, pooled over the other nine people, causal recentring) are combined per new session
with weights from their causal agreement with the chain (label-free), against equal weights and an oracle (weights from
true accuracy so far); fused into the routed chain (beta 0.10). Usage: python studies/12nov/bandsel.py [smoke]"""
import os, sys
import numpy as np
REPO = os.path.abspath(__file__ + "/../../..")
sys.path[:0] = [REPO + "/replay", REPO + "/studies/12nov"]
SMOKE = "smoke" in sys.argv
src = open(REPO + "/studies/12nov/bandmap.py").read(); exec(compile(src.split("keys = [")[0], "bandmap", "exec"))  # noqa: S102  (O, fit, sess, PEOPLE, NB)
import env  # noqa: E402
import grid as G  # noqa: E402
A = env.A; p = "all_loso"; Z = A.L.Z; S = Z[f"sess|{p}"]; y = O.y.astype(int); N = len(y); rng = np.random.default_rng(0)


def probs(M, prior, r):
    Cw, _ = O.recentred(r, prior); out = []
    for b in range(NB):
        out.append(M["T", b].predict_proba(O.tang([Cw[b]]))); W, m = M["C", b]; out.append(m.predict_proba(np.log(np.einsum("cf,ncd,df->nf", W, Cw[b], W))))
    return np.stack(out)  # (2*NB, n, 3)


BP = np.full((2 * NB, N, 3), 1 / 3); PEOPLE_ = PEOPLE[:2] if SMOKE else PEOPLE
for t in PEOPLE_:
    M, pr = fit([r for q in PEOPLE if q != t for r in sess(q)])
    for r in sess(t):
        BP[:, r] = probs(M, pr, r)
    print("fold", t, flush=True)
C0 = A.system("B2_pool", A.FAMS)["out"][p]; lb = np.log(np.clip(BP, 1e-3, 1)); out = {k: np.zeros((N, 3)) for k in ("equal", "agree", "oracle")}
for s in np.unique(S):
    r = np.flatnonzero(S == s)
    for i, t in enumerate(r):
        past = r[:i]
        if len(past) < 10:
            w = {k: np.ones(2 * NB) for k in out}
        else:
            ag = (BP[:, past].argmax(2) == C0[past].argmax(1)).mean(1); ac = (BP[:, past].argmax(2) == y[past]).mean(1)
            w = {"equal": np.ones(2 * NB), "agree": np.exp(8 * (ag - ag.max())), "oracle": np.exp(8 * (ac - ac.max()))}
        for k in out:
            L = (w[k][:, None] * lb[:, t]).sum(0) / w[k].sum(); e = np.exp(L - L.max()); out[k][t] = e / e.sum()
m = np.isin(O.PER, PEOPLE_)
print("alone, window accuracy: " + " | ".join(f"{k} {100 * np.mean(v[m].argmax(1) == y[m]):.1f}" for k, v in out.items()), flush=True)
for bn in ("B2_pool",) if SMOKE else env.BASES:
    C = A.system(bn, A.FAMS)["out"][p]; ref = G.scores(C, p)
    for k, P in out.items():
        L = np.log(np.clip(C, 1e-9, 1)) + 0.10 * np.log(np.clip(P, 1e-3, 1)); e = np.exp(L - L.max(1, keepdims=True)); d = G.scores(e / e.sum(1, keepdims=True), p) - ref
        bt = np.percentile([d[rng.integers(0, 10, 10)].mean() for _ in range(4000)], [10, 90]); print(f"  {bn} band ensemble {k:6s} fused: {d.mean():+.2f} [{bt[0]:+.2f}, {bt[1]:+.2f}]", flush=True)
