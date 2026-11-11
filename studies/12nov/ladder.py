"""LADDER (ledger s35): causal in-session logistic head on true vs chain labels, after the full v3 chain.
Usage: python studies/12nov/ladder.py [BASE,BASE]"""
import os, sys, warnings
import numpy as np
from sklearn.linear_model import LogisticRegression
REPO = os.path.abspath(__file__ + "/../../..")
sys.path[:0] = [REPO + "/replay"]
import env  # noqa: E402
A = env.A; SC = A.SC; Z = A.L.Z
warnings.filterwarnings("ignore")
T0, STEP, C = 20, 5, 0.1


def head(P, proto, labels):
    """-> head posterior (n,3), NaN where no head yet. labels: 'true' or 'chain'."""
    ss, X, y = Z[f"sess|{proto}"], Z[f"pc128|{proto}"], Z[f"y|{proto}"]; H = np.full_like(P, np.nan)
    for s in np.unique(ss):
        r = np.flatnonzero(ss == s); lab = y[r] if labels == "true" else P[r].argmax(1); w = np.ones(len(r)) if labels == "true" else P[r].max(1)
        for t in range(T0, len(r), STEP):
            if len(np.unique(lab[:t])) < 3:
                continue
            m, sd = X[r[:t]].mean(0), np.maximum(X[r[:t]].std(0), 1e-3)
            clf = LogisticRegression(C=C, max_iter=300).fit((X[r[:t]] - m) / sd, lab[:t], sample_weight=w[:t])
            q = r[t:t + STEP]; H[q] = clf.predict_proba((X[q] - m) / sd)
    return H


def fuse(P, H, proto, lam):
    ss = Z[f"sess|{proto}"]; t = np.zeros(len(P))
    for s in np.unique(ss):
        r = np.flatnonzero(ss == s); t[r] = np.arange(len(r))
    out = P.copy(); m = ~np.isnan(H[:, 0]); e = lam * t[m] / (t[m] + 60)
    q = P[m] * np.clip(H[m], 1e-6, 1) ** e[:, None]; out[m] = q / q.sum(1, keepdims=True); return out


for bn in (sys.argv[1].split(",") if len(sys.argv) > 1 else env.EXPL):
    base = A.system(bn, A.FAMS)["out"]; print(SC.absline(bn + " chain", base), flush=True)
    for labels in ("true", "chain"):
        H = {p: head(base[p], p, labels) for p in A.PROTOS}
        ha = {p: 100 * np.mean(np.nanargmax(np.nan_to_num(H[p], nan=-1), 1)[~np.isnan(H[p][:, 0])] == Z[f"y|{p}"][~np.isnan(H[p][:, 0])]) for p in A.PROTOS}
        print(f"   head alone ({labels} labels): window accuracy new {ha['all_loso']:.1f} seen {ha['seen_session']:.1f}")
        for lam in (0.5, 1.0):
            arm = {p: fuse(base[p], H[p], p, lam) for p in A.PROTOS}
            print("   " + SC.line(f"{labels} labels, lam {lam}", SC.compare(arm, base)), flush=True)
