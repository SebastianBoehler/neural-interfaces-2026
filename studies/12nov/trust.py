"""TRUST (ledger s79) diagnostic: best beta per returning session against label-free signals of the personal members.
Usage: python studies/12nov/trust.py"""
import os, sys
import numpy as np
from scipy.stats import spearmanr
REPO = os.path.abspath(__file__ + "/../../..")
F = REPO + "/studies/12nov/v5_replay.py"; exec(compile(open(F).read().split("\nres = []")[0], F, "exec"))  # noqa: S102
BETAS = (0.0, 0.05, 0.10, 0.15, 0.25, 0.40); lg = lambda P: np.log(np.clip(P, 1e-3, 1))  # noqa: E731
rows = lambda p, s: np.flatnonzero((O.PER == p) & (O.SES == s))  # noqa: E731


def signals(m, chain):
    a = [m[k].argmax(1) for k in "TCS"]; mm = sum(lg(m[k]) for k in "TCS") / 3
    return dict(tcs_agree=np.mean([np.mean(a[i] == a[j]) for i, j in ((0, 1), (0, 2), (1, 2))]), chain_agree=np.mean(mm.argmax(1) == chain.argmax(1)),
                conf=np.mean(np.exp(mm - np.log(np.exp(mm).sum(1, keepdims=True))).max(1)))


def stage(P, m, b):
    L = np.log(np.clip(P, 1e-9, 1)) + b * sum(lg(m[k]) for k in "TCS") / 3; e = np.exp(L - L.max(1, keepdims=True)); return e / e.sum(1, keepdims=True)


print("real seen sessions (routed chain, ramp ignored: the stage applied to the whole session)")
for bn in ("B2_pool", "B2_s345", "B2_s012"):
    REC.clear(); out0, _ = ns["system4"](bn, "max", 10); P = out0["seen_session"]; idx = A.pl.B["idx|seen_session"]; pos = {int(g): k for k, g in enumerate(idx)}
    for q in ("P04", "P10"):
        for sn in "23":
            g = rows(q, sn); r = np.array([pos[int(x)] for x in g]); m = O.Personal([rows(q, "1")]).stream(g)
            acc = [O.ba(g, stage(P[r], m, b)) for b in BETAS]; sg = signals(m, P[r])
            print(f"  {bn} {q} s{sn}: chain {acc[0]:.1f} | gain by beta " + " ".join(f"{b}:{a - acc[0]:+.1f}" for b, a in zip(BETAS, acc)) + " | " + " ".join(f"{k} {v:.2f}" for k, v in sg.items()), flush=True)
print("12 cross-session pairs (LOSO chain)")
base = A.system("B2_pool", A.FAMS)["out"]["all_loso"]; X = []
for q in ("P04", "P10"):
    for a in "123":
        for b_ in "123":
            if a == b_:
                continue
            r = rows(q, b_); m = O.Personal([rows(q, a)]).stream(r); acc = [O.ba(r, stage(base[r], m, b)) for b in BETAS]; sg = signals(m, base[r])
            best = BETAS[int(np.argmax(acc))]; X.append((best, acc[3] - acc[0], sg))
            print(f"  {q} s{a}->s{b_}: chain {acc[0]:.1f} | best beta {best} | gain@0.15 {acc[3] - acc[0]:+.1f} | " + " ".join(f"{k} {v:.2f}" for k, v in sg.items()), flush=True)
for k in X[0][2]:
    print(f"spearman({k}, best beta) {spearmanr([x[2][k] for x in X], [x[0] for x in X])[0]:+.2f} | ({k}, gain at 0.15) {spearmanr([x[2][k] for x in X], [x[1] for x in X])[0]:+.2f}")
