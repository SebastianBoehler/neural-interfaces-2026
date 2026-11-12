"""SEQCAL (ledger s55): Graz cue-order prior with a learned fusion, true interface and run position (ceiling).
Usage: python studies/12nov/seqcal.py"""
import os, sys, warnings
import numpy as np
from sklearn.linear_model import LogisticRegression
REPO = os.path.abspath(__file__ + "/../../..")
sys.path[:0] = [REPO + "/replay"]
import env  # noqa: E402
warnings.filterwarnings("ignore")
A = env.A; SC = A.SC; L = A.L; Z = L.Z
PAT = np.array([2, 1, 0])  # WORD, SUB, MI by run position modulo 3


def feats(P, proto):
    idx = Z[f"idx|{proto}"]; graz = L.IFACE[idx] == "Graz"; pos = L.runpos(proto); oh = np.eye(3)[PAT[pos % 3]]
    return np.c_[np.log(np.clip(P, 1e-6, 1)), oh], graz


for bn in env.EXPL:
    base = A.system(bn, A.FAMS)["out"]; Xn, gn = feats(base["all_loso"], "all_loso"); Xs, gs = feats(base["seen_session"], "seen_session")
    yn, pn = Z["y|all_loso"], Z["person|all_loso"]; ps = Z["person|seen_session"]; out = {p: v.copy() for p, v in base.items()}
    for p in np.unique(pn):
        tr = gn & (pn != p); m = LogisticRegression(C=1.0, max_iter=1000).fit(Xn[tr], yn[tr])
        te = gn & (pn == p); out["all_loso"][te] = m.predict_proba(Xn[te])
        te = gs & (ps == p)
        if te.any():
            out["seen_session"][te] = m.predict_proba(Xs[te])
    r = SC.compare(out, base); acc = lambda P: 100 * np.mean(P[gn].argmax(1) == yn[gn])  # noqa: E731
    print(SC.line(f"{bn} learned fusion", r)); print(f"   Graz window accuracy, new people: {acc(base['all_loso']):.2f} -> {acc(out['all_loso']):.2f}; prior alone {100 * np.mean(PAT[L.runpos('all_loso') % 3][gn] == yn[gn]):.1f}", flush=True)
