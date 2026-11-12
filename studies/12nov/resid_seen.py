"""RESID2 (ledger s85): stacked personal correction in the v5.1 replay, seen protocol, six bases. Usage: python studies/12nov/resid_seen.py"""
import os, sys
import numpy as np
from sklearn.linear_model import LogisticRegression
REPO = os.path.abspath(__file__ + "/../../..")
F = REPO + "/studies/12nov/v5_replay.py"; exec(compile(open(F).read().split("\nres = []")[0], F, "exec"))  # noqa: S102
RUN = np.load(REPO + "/data/proteus_windows.npz", allow_pickle=True)["run"].astype(int); lg = lambda P: np.log(np.clip(P, 1e-3, 1))  # noqa: E731
W = float(os.environ.get("W", 0.5))


def feats(Pc, m):
    L = [lg(Pc)] + [lg(m[k]) for k in "TCS"]; return np.hstack(L + [L[0] * (L[1] + L[2] + L[3]) / 3])


CVM = {}
for q in ("P04", "P10"):
    a = np.flatnonzero((O.PER == q) & (O.SES == "1")); out = {k: np.zeros((len(a), 3)) for k in "TCS"}
    for r in sorted(set(RUN[a])):
        te, tr = a[RUN[a] == r], a[RUN[a] != r]; o = O.Personal([tr]).stream(te)
        for k in "TCS":
            out[k][RUN[a] == r] = o[k]
    CVM[q] = (a, out)


def staged_resid(out0, clf):
    out = {p: v.copy() for p, v in out0.items()}
    for (proto, s), (r, j) in REC.items():
        names, gi, f = NM[(proto, s)]; idx = A.pl.B[f"idx|{proto}"]; pos = {int(g): k for k, g in enumerate(idx)}; rows = np.array([pos[int(g)] for g in gi])
        on = np.flatnonzero(r > 0)
        for t in on:
            q = names[int(j[t])]; m = members(proto, s, q)
            if q[-3:] not in clf:
                continue
            R = clf[q[-3:]].predict_proba(feats(out0[proto][rows[t]][None], {k: m[k][t][None] for k in "TCS"}))[0]
            L = np.log(np.clip(out0[proto][rows[t]], 1e-9, 1)) + W * r[t] * np.log(np.clip(R, 1e-3, 1)); e = np.exp(L - L.max()); out[proto][rows[t]] = e / e.sum()
    return out


res = []
for bn in ("B2_pool", "B2_s345", "B2_s012", "B2_s0", "B2_s1", "B2_s2"):
    loso = A.system(bn, A.FAMS)["out"]["all_loso"]; clf = {}
    for q, (a, m) in CVM.items():
        clf[q] = LogisticRegression(C=0.1, max_iter=2000).fit(feats(loso[a], m), O.y[a])
    REC.clear(); out0, _ = ns["system4"](bn, "max", 10); v51, _ = staged(out0); rs = staged_resid(out0, clf); r = SC.compare(rs, v51); res.append(r)
    st = SC.compare(v51, out0)
    print(f"{bn}: v5.1 stage seen {st['seen']:+.2f} (P04 {st['ds'][0]:+.2f}, P10 {st['ds'][1]:+.2f}) | RESID w {W} against v5.1: seen {r['seen']:+.2f} (P04 {r['ds'][0]:+.2f}, P10 {r['ds'][1]:+.2f}) new {r['new']:+.3f}", flush=True)
ok = sum(r["seen"] >= 0.5 and r["ds"][1] >= -0.5 for r in res)
print(f"MEAN against v5.1: seen {np.mean([r['seen'] for r in res]):+.2f} (P04 {np.mean([r['ds'][0] for r in res]):+.2f}, P10 {np.mean([r['ds'][1] for r in res]):+.2f}) new {np.mean([r['new'] for r in res]):+.3f}; bases meeting the lever {ok}/6")
