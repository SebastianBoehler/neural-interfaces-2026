"""RESID (ledger s84): stacked personal correction of the chain, on the INPOP proxy (runs 1-3 public, 4-6 later).
Usage: python studies/12nov/inpop2.py"""
import os, sys
import numpy as np
from sklearn.linear_model import LogisticRegression
F = os.path.abspath(__file__ + "/../inpop.py"); exec(compile(open(F).read().split("\nMEM = {}")[0], F, "exec"))  # noqa: S102
lg = lambda P: np.log(np.clip(P, 1e-3, 1))  # noqa: E731


def chain_rows(bn, rows, inpop):
    global B, RB, sess  # reuse inpop.chain on another row set (it reads the module globals B, RB, sess)
    B0, RB0, s0 = B, RB, sess; B = rows; sess = np.array([f"{O.PER[i]}|{'late' if RUN[i] >= 4 else 'early'}" for i in rows])
    try:
        assert not inpop; return chain(bn, False)
    finally:
        B, RB, sess = B0, RB0, s0


PUB = {q: np.flatnonzero((O.PER == q) & (O.SES == "1") & (RUN <= 3)) for q in PEOPLE}; LATE = {q: B[O.PER[B] == q] for q in PEOPLE}
CV = {}
for q in PEOPLE:  # personal members on the public runs, cross-fitted run by run
    a = PUB[q]; out = {k: np.zeros((len(a), 3)) for k in "TCS"}
    for r in (1, 2, 3):
        te = a[RUN[a] == r]; tr = a[RUN[a] != r]; o = O.Personal([tr]).stream(te); k_ = RUN[a] == r
        for k in "TCS":
            out[k][k_] = o[k]
    CV[q] = (out, O.Personal([a]).stream(LATE[q]))


def ba_people(P):
    return np.array([O.ba(LATE[q], P[O.PER[B] == q]) for q in PEOPLE])


def feats(Pc, m):
    L = [lg(Pc)] + [lg(m[k]) for k in "TCS"]; return np.hstack(L + [L[0] * (L[1] + L[2] + L[3]) / 3])


for bn in ("B2_pool", "B2_s345", "B2_s012"):
    pub_chain = chain_rows(bn, np.concatenate([PUB[q] for q in PEOPLE]), False); allpub = np.concatenate([PUB[q] for q in PEOPLE])
    Pl = chain(bn, True); ref = ba_people(Pl); stg = Pl.copy(); res = {w: Pl.copy() for w in (0.5, 1.0)}
    for q in PEOPLE:
        k = O.PER[B] == q; mcv, mlate = CV[q]; L = lg(Pl[k]) + 0.10 * sum(lg(mlate[x]) for x in "TCS") / 3; e = np.exp(L - L.max(1, keepdims=True)); stg[k] = e / e.sum(1, keepdims=True)
        pc = pub_chain[np.isin(allpub, PUB[q])]; clf = LogisticRegression(C=0.1, max_iter=2000).fit(feats(pc, mcv), O.y[PUB[q]])
        R = clf.predict_proba(feats(Pl[k], mlate))
        for w in res:
            L = lg(Pl[k]) + w * lg(R); e = np.exp(L - L.max(1, keepdims=True)); res[w][k] = e / e.sum(1, keepdims=True)
    ds = ba_people(stg) - ref; print(f"{bn}: in-population chain {ref.mean():.1f} | v5.1 stage {ds.mean():+.2f} | " + " ".join(f"{x:+.1f}" for x in ds), flush=True)
    for w, P in res.items():
        d = ba_people(P) - ref; bt = np.percentile([(d - ds)[rng.integers(0, 10, 10)].mean() for _ in range(4000)], [10, 90])
        print(f"   RESID w {w}: {d.mean():+.2f} | against the stage {(d - ds).mean():+.2f} [{bt[0]:+.2f}, {bt[1]:+.2f}], ahead {(d > ds).sum()}/10 | " + " ".join(f"{x:+.1f}" for x in d), flush=True)
