"""CLUSTX (ledger s110): own-space experts for person types (cluster of the training people) against the pooled nine,
oracle and label-free choice per new session, fused as a final stage. Usage: python studies/12nov/clustx.py"""
import os, sys
import numpy as np
REPO = os.path.abspath(__file__ + "/../../..")
sys.path[:0] = [REPO + "/replay", REPO + "/studies/12nov"]
import env  # noqa: E402
import ownspace as O  # noqa: E402
import grid as G  # noqa: E402
A = env.A; p = "all_loso"; Z = A.L.Z; sess = Z[f"sess|{p}"]; y = O.y.astype(int); N = len(y); rng = np.random.default_rng(0)
assert (Z[f"idx|{p}"] == np.arange(N)).all()
TYPES = {"A": {"P01", "P02", "P07", "P08"}, "B": {"P03", "P06", "P09"}}; PEOPLE = sorted(set(O.PER))
rows_of = lambda q: [np.flatnonzero((O.PER == q) & (O.SES == s)) for s in sorted(set(O.SES[O.PER == q]))]  # noqa: E731
lg = lambda P: np.log(np.clip(P, 1e-3, 1))  # noqa: E731
EXP = {k: np.full((N, 3), 1 / 3) for k in ("pool", "A", "B")}
for t in PEOPLE:
    groups = {"pool": [q for q in PEOPLE if q != t], "A": [q for q in TYPES["A"] if q != t], "B": [q for q in TYPES["B"] if q != t]}
    for k, g in groups.items():
        m = O.Personal([r for q in g for r in rows_of(q)])
        for r in rows_of(t):
            o = m.stream(r); L = sum(lg(o[x]) for x in "TCS"); e = np.exp(L - L.max(1, keepdims=True)); EXP[k][r] = e / e.sum(1, keepdims=True)
    print("done", t, flush=True)
C0 = A.system("B2_pool", A.FAMS)["out"][p]; ORA, LF = np.zeros(N, int), np.zeros(N, int); keys = list(EXP)
for s in np.unique(sess):
    r = np.flatnonzero(sess == s); acc = [np.mean(EXP[k][r].argmax(1) == y[r]) for k in keys]; agr = [np.mean(EXP[k][r].argmax(1) == C0[r].argmax(1)) for k in keys]
    ORA[r], LF[r] = int(np.argmax(acc)), int(np.argmax(agr))
pick = lambda sel: np.stack([EXP[keys[k]][i] for i, k in enumerate(sel)])  # noqa: E731
print(f"alone, window accuracy: " + " | ".join(f"{k} {100 * np.mean(EXP[k].argmax(1) == y):.1f}" for k in keys) + f" | oracle pick {100 * np.mean(pick(ORA).argmax(1) == y):.1f} | label-free pick {100 * np.mean(pick(LF).argmax(1) == y):.1f}")
for bn in env.BASES:
    C = A.system(bn, A.FAMS)["out"][p]; ref = G.scores(C, p)
    for tag, P in (("pooled expert", EXP["pool"]), ("label-free pick", pick(LF)), ("oracle pick", pick(ORA))):
        L = np.log(np.clip(C, 1e-9, 1)) + 0.10 * lg(P); e = np.exp(L - L.max(1, keepdims=True)); d = G.scores(e / e.sum(1, keepdims=True), p) - ref
        bt = np.percentile([d[rng.integers(0, 10, 10)].mean() for _ in range(4000)], [10, 90]); print(f"  {bn} {tag:15s}: {d.mean():+.2f} [{bt[0]:+.2f}, {bt[1]:+.2f}]", flush=True)
