"""PCURVE (ledger s146): the chain-level people curve, reduced form. A core1 surrogate (balanced logistic regression on
core1's 256 exact-SVD components, per left-out-person fold; agrees with core1 on about 82 % of windows) is trained on
k = 3, 5, 7 or 9 of the nine training people (five random draws below 9), and scored alone (core level) and through
the shipped stages as the only member (gm.pipe_g: rebalancing, gated SessionEM, gated TSEM). New people only.
Limit: the PC basis of each fold was computed on all nine training people (unlabelled features of the unused people
enter the basis, not the classifier). Usage: [DRAWS=5] python studies/13nov/peoplecurve.py"""
import glob, os, sys
import numpy as np
from sklearn.linear_model import LogisticRegression
REPO = os.path.abspath(__file__ + "/../../..")
sys.path.insert(0, REPO + "/replay")
import env  # noqa: E402
A = env.A; SC = A.SC; Z = A.L.Z; p = "all_loso"; N = len(Z[f"y|{p}"]); per = Z[f"person|{p}"]; gm = A.gm
DRAWS = int(os.environ.get("DRAWS", 5)); KS = (3, 5, 7, 9); rng = np.random.default_rng(0)
FOLD = {}
for f in sorted(glob.glob(f"{REPO}/runs/replay_data/feats_exact/all_loso-P*.npz")):
    z = np.load(f, allow_pickle=True); q = os.path.basename(f)[9:12]; itr = z["idx_tr"].astype(int)
    te = [(z[k].astype(np.float64), z[k.replace("|pc", "|idx")].astype(int)) for k in z.files if k.endswith("|pc")]
    FOLD[q] = (z["pc_tr"].astype(np.float64), z["y_tr"].astype(int), per[itr], te)
assert sorted(FOLD) == list(SC.PEOPLE), sorted(FOLD)
print(f"PCURVE: k {KS}, draws {DRAWS}; core1 surrogate (LR on 256 PCs, C 0.01, standardised) alone and through gm.pipe_g", flush=True)


def predict(k, d):
    P = np.zeros((N, 3)); r = np.random.default_rng(1000 * k + d)
    for q, (X, y, pp, te) in FOLD.items():
        others = sorted(set(pp)); use = others if k == len(others) else list(r.choice(others, k, replace=False))
        m = np.isin(pp, use); sd = X[m].std(0) + 1e-9
        lr = LogisticRegression(C=0.01, class_weight="balanced", max_iter=3000).fit(X[m] / sd, y[m])
        for Xt, idx in te:
            P[idx] = lr.predict_proba(Xt / sd)
    assert np.abs(P.sum(1) - 1).max() < 1e-6, "a row was not predicted"
    return P


R = {}
for k in KS:
    for d in range(1 if k == 9 else DRAWS):
        P = predict(k, d); core = SC.person(P, p); chain = SC.person(gm.pipe_g(P, p), p); R[k, d] = (core, chain)
        print(f"k {k} draw {d}: core {core.mean():.2f} | chain {chain.mean():.2f} (P04 {chain[3]:.1f})", flush=True)
print("\nk | core mean [range over draws] | chain mean [range] | stage gain")
M = {}
for k in KS:
    c = np.array([R[kk, d][0].mean() for kk, d in R if kk == k]); h = np.array([R[kk, d][1].mean() for kk, d in R if kk == k]); M[k] = (c.mean(), h.mean())
    print(f"{k} | {c.mean():.2f} [{c.min():.2f}, {c.max():.2f}] | {h.mean():.2f} [{h.min():.2f}, {h.max():.2f}] | {h.mean() - c.mean():+.2f}", flush=True)
x = np.log2(np.array(KS, float))
for i, name in ((0, "core"), (1, "chain")):
    y = np.array([M[k][i] for k in KS]); b = np.polyfit(x, y, 1)[0]; b2 = np.polyfit(x[1:], y[1:], 1)[0]
    print(f"{name} slope per doubling of people: {b:+.2f} over k 3-9, {b2:+.2f} over k 5-9", flush=True)
