"""TEACH3 (ledger s148): score the teachers retrained on Leonardo on 13 Nov, new people, six bases, A.system.
  m7c2  the seven m7 corpora again (control: the rebuild must reproduce the stored m7c teacher, runs/tsch/heads.npz)
  cog1  a cognitive teacher on Shin2017B only (subtraction vs rest), as an extra member next to the reference teacher
Teachers are seed means of both trunks mapped per left-out person (balanced LR, C 0.01), as teach2.py. Contexts: the
shipped chain and FT+OCU+REST (s140's design input). Seen keeps the shipped teacher.
Usage: python studies/13nov/teach3.py"""
import os, sys
import numpy as np
from sklearn.linear_model import LogisticRegression
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stacklib import A, BASES, REPO, Z, rng, run  # noqa: E402
SC = A.SC; p0 = "all_loso"; per = Z[f"person|{p0}"]; y = Z[f"y|{p0}"].astype(int); N = len(y); SHIP = Z[f"B2_pool|Ps|seen_session"][10]


def heads(path, tag, seeds=range(12)):
    H = np.load(path); out = []
    for trunk in ("eegnex", "ifnet"):
        ks = [f"{trunk}_ea-{tag}-s{s}" for s in seeds if f"{trunk}_ea-{tag}-s{s}" in H.files]
        assert len(ks) == len(seeds) or tag == "m7c", (path, tag, trunk, len(ks))
        X = np.mean([H[k] for k in ks], 0); assert X.shape[0] == N and X.max() <= 1e-9; out.append(X)
    return np.hstack(out)


def teacher(X):
    P = np.zeros((N, 3))
    for q in np.unique(per):
        tr, te = per != q, per == q
        P[te] = LogisticRegression(C=0.01, class_weight="balanced", max_iter=3000).fit(X[tr], y[tr]).predict_proba(X[te])
    return P


T = {"t7c": teacher(heads(REPO + "/runs/tsch/heads.npz", "m7c", range(6))), "m7c2": teacher(heads(REPO + "/runs/teach/heads_m7c2.npz", "m7c2")),
     "cog1": teacher(heads(REPO + "/runs/teach/heads_cog1.npz", "cog1"))}
for k, P in T.items():
    A.EXTRA[k] = {p0: P, "seen_session": SHIP}; print(f"{k:5s} alone new {SC.person(P, p0).mean():.2f}", flush=True)
ARMS = {"ref": dict(teach=("t7c", 5.0)), "m7c2": dict(teach=("m7c2", 5.0)), "cog1_w1": dict(teach=("t7c", 5.0), add=[("cog1", 1.0)]),
        "cog1_w2": dict(teach=("t7c", 5.0), add=[("cog1", 2.0)]), "cog1_w05": dict(teach=("t7c", 5.0), add=[("cog1", 0.5)])}
CTX = {"shipped": ((), True), "FT+OCU+REST": (("FT", "OCU", "REST"), False)}
R = {}
for bn in BASES:
    for c, (S, sh) in CTX.items():
        for k, kw in ARMS.items():
            R[bn, c, k] = run(S, bn, shipped=sh, **kw)[0]
    print(bn + " " + " | ".join(f"{c}: " + " ".join(f"{k} {R[bn, c, k].mean() - R[bn, c, 'ref'].mean():+.2f}" for k in ARMS if k != "ref") for c in CTX), flush=True)
for c in CTX:
    print(f"\n{c}: against the stored m7c teacher at weight 5, new, six-base mean")
    for k in ARMS:
        if k == "ref":
            continue
        d = np.mean([R[b, c, k] - R[b, c, "ref"] for b in BASES], 0); bt = np.percentile([d[rng.integers(0, 10, 10)].mean() for _ in range(4000)], [10, 90]); q = int(np.argmax(d))
        print(f"  {k:9s} new {d.mean():+.2f} [{bt[0]:+.2f}, {bt[1]:+.2f}] people>=0 {(d >= 0).sum()}/10, without P{q + 1:02d} {np.delete(d, q).mean():+.2f} | P04 {d[3]:+.2f} P10 {d[9]:+.2f}", flush=True)
