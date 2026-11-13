"""TEACH2 (ledger s132, gap 2 of docs/SOLUTION_SPACE_MAP.md): the external teacher line redesigned from its stored head
outputs (runs/tsch/heads.npz: EEGNeX + IFNet trunks, m7c control / m8s with Scherer, seeds 0-5) and re-scored in two
contexts, the shipped chain and the STACK design (FT+OCU), new people, six bases. Teacher maps are fitted per left-out
person on the other nine (balanced logistic regression, C 0.01). Seen sessions keep the shipped teacher.
Usage: [BASES=B2_pool] [SMOKE=1] python studies/13nov/teach2.py"""
import os, sys, time
import numpy as np
from sklearn.linear_model import LogisticRegression
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stacklib import A, BASES, REPO, SMOKE, Z, rng, run  # noqa: E402
SC = A.SC; p0 = "all_loso"; sess = Z[f"sess|{p0}"]; per = Z[f"person|{p0}"]; y = Z[f"y|{p0}"].astype(int); N = len(y)
H = np.load(REPO + "/runs/tsch/heads.npz"); NAMES = [str(x) for x in H["names"]]; SHIP = {p: Z[f"B2_pool|Ps|{p}"][10] for p in A.PROTOS}
COG = ("physionetmi:rest", "schirrmeister2017:rest", "shin2017b:rest", "shin2017b:subtraction", "scherer2015t:subtraction", "scherer2015t:word")


def heads(tag, cols=None):
    out = []
    for trunk in ("eegnex", "ifnet"):
        X = np.mean([H[f"{trunk}_ea-{tag}-s{s}"] for s in range(6)], 0); names = NAMES[:X.shape[1]]
        assert X.shape == (N, len(names)) and X.max() <= 1e-9, (tag, X.shape)
        out.append(X if cols is None else X[:, [names.index(c) for c in cols if c in names]])
    return np.hstack(out)


def sessnorm(X):  # causal: each row minus the mean of its session's rows up to and including it
    out = np.empty_like(X)
    for s in np.unique(sess):
        r = np.flatnonzero(sess == s); out[r] = X[r] - np.cumsum(X[r], 0) / np.arange(1, len(r) + 1)[:, None]
    return out


def teacher(X):
    P = np.zeros((N, 3))
    for q in np.unique(per):
        tr, te = per != q, per == q
        P[te] = LogisticRegression(C=0.01, class_weight="balanced", max_iter=3000).fit(X[tr], y[tr]).predict_proba(X[te])
    return P


T = {"t7c": teacher(heads("m7c")), "t8s": teacher(heads("m8s")), "n7c": teacher(sessnorm(heads("m7c"))),
     "n8s": teacher(sessnorm(heads("m8s"))), "cog7": teacher(heads("m7c", COG)), "cog8": teacher(heads("m8s", COG))}
assert heads("m8s", COG).shape[1] > heads("m7c", COG).shape[1], "the Scherer heads were not found in m8s"
for k, P in T.items():
    A.EXTRA[k] = {p0: P, "seen_session": SHIP["seen_session"]}
    print(f"{k:5s} alone new {SC.person(P, p0).mean():.2f}", flush=True)
alone = SC.person(T["t7c"], p0).mean(); assert abs(alone - 50.46) < 1.5, f"rebuilt m7c teacher {alone:.2f}, 11 Nov harness 50.46"
print(f"shipped teacher alone new {SC.person(SHIP[p0], p0).mean():.2f}", flush=True)

ARMS = {"shipped": dict(), "ref": dict(teach=("t7c", 5.0)), "t8s_w2": dict(teach=("t8s", 2.0)), "t8s_w5": dict(teach=("t8s", 5.0)),
        "n7c_w5": dict(teach=("n7c", 5.0)), "n8s_w2": dict(teach=("n8s", 2.0)), "n8s_w5": dict(teach=("n8s", 5.0)),
        "cog7_w1": dict(teach=("t7c", 5.0), add=[("cog7", 1.0)]), "cog7_w2": dict(teach=("t7c", 5.0), add=[("cog7", 2.0)]),
        "cog8_w1": dict(teach=("t7c", 5.0), add=[("cog8", 1.0)]), "cog8_w2": dict(teach=("t7c", 5.0), add=[("cog8", 2.0)])}
CTX = {"shipped": ((), True), "FT+OCU": (("FT", "OCU"), False)}
if SMOKE:
    ARMS = {k: ARMS[k] for k in ("shipped", "ref", "n8s_w5")}; BASES = BASES[:1]
R = {}
for bn in BASES:
    for c, (S, sh) in CTX.items():
        t0 = time.time()
        for k, kw in ARMS.items():
            R[bn, c, k] = run(S, bn, shipped=sh, **kw)
        b = R[bn, c, "ref"]
        print(f"{bn} {c} ({time.time() - t0:.0f}s): ref new {b[0].mean():.2f} | " + " ".join(f"{k} {R[bn, c, k][0].mean() - b[0].mean():+.2f}/{R[bn, c, k][1].mean() - b[1].mean():+.2f}" for k in ARMS if k != "ref"), flush=True)
for c in CTX:
    print(f"\n{c} context: against the rebuilt m7c teacher at weight 5, new people, mean over bases")
    for k in ARMS:
        if k == "ref":
            continue
        dn = np.array([R[bn, c, k][0] - R[bn, c, "ref"][0] for bn in BASES]); ds = np.mean([(R[bn, c, k][1] - R[bn, c, "ref"][1]).mean() for bn in BASES]); d = dn.mean(0)
        bt = np.percentile([d[rng.integers(0, 10, 10)].mean() for _ in range(4000)], [10, 90])
        print(f"  {k:9s} new {d.mean():+.2f} [{bt[0]:+.2f}, {bt[1]:+.2f}] bases +{(dn.mean(1) > 0).sum()}/{len(BASES)} ({dn.mean(1).min():+.2f} to {dn.mean(1).max():+.2f}) people>=0 {(d >= 0).sum()}/10 P04 {d[3]:+.2f} P10 {d[9]:+.2f} | seen {ds:+.2f}", flush=True)
