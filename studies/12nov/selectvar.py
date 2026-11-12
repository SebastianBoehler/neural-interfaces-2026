"""SELECT (ledger s76): label-free per-session choice among chain variants (new people). Usage: python studies/12nov/selectvar.py"""
import glob, os, sys
import numpy as np
REPO = os.path.abspath(__file__ + "/../../..")
sys.path.insert(0, REPO + "/replay")
import env  # noqa: E402
import grid as G  # noqa: E402
A = env.A; p = "all_loso"; Z = A.L.Z; sess = Z[f"sess|{p}"]; y = Z[f"y|{p}"].astype(int); per = Z[f"person|{p}"]; X = Z[f"pc128|{p}"]; rng = np.random.default_rng(1)
cell = Z[f"cell|{p}"]; N = len(y); PC = np.zeros((N, 256))
for f in sorted(glob.glob(f"{REPO}/runs/replay_data/feats_exact/all_loso-P*.npz")):
    z = np.load(f, allow_pickle=True); sd = z["pc_tr"].std(0)
    for k in [x for x in z.files if x.endswith("|pc")]:
        PC[z[k.replace("|pc", "|idx")].astype(int)] = z[k] / sd
W = [np.linalg.qr(rng.standard_normal((256, 64)))[0] for _ in range(2)]


def crit(P, r, maj):
    lab = P[r].argmax(1); c = np.bincount(lab, minlength=3) / len(r); ent = -(c[c > 0] * np.log(c[c > 0])).sum()
    Xs = (X[r] - X[r].mean(0)) / np.maximum(X[r].std(0), 1e-6); mu = np.stack([Xs[lab == k].mean(0) if (lab == k).sum() > 1 else np.zeros(Xs.shape[1]) for k in range(3)])
    fis = (mu ** 2).sum(1).mean() / max(1e-9, np.mean([Xs[lab == k].var(0).sum() for k in range(3) if (lab == k).sum() > 1]))
    return dict(conf=P[r].max(1).mean(), balance=ent, agree=np.mean(lab == maj[r]), fisher=fis)


for bn in env.EXPL:
    V = {}
    for lam in (0.12, 0.15, 0.20, 0.25, 0.30):
        A.LAM = lam; V[f"lam{lam}"] = A.system(bn, A.FAMS)["out"][p]
    A.LAM = 0.15; base = V["lam0.15"]
    for i, w in enumerate(W):
        L, tt = G.tsem_lik(base, sess, PC @ w); V[f"rand{i}"] = G.apply_lam(base, L, tt, 0.15)
    names = list(V); maj = np.stack([V[k].argmax(1) for k in names]); maj = np.array([np.bincount(c, minlength=3).argmax() for c in maj.T])
    ref = G.scores(base, p); pick = {c: base.copy() for c in ("oracle", "conf", "balance", "agree", "fisher")}
    for s in np.unique(sess):
        r = np.flatnonzero(sess == s); ba = {k: np.mean([np.mean(V[k][r][(cell[r] == c) & (y[r] == j)].argmax(1) == j) for c in np.unique(cell[r]) for j in range(3) if ((cell[r] == c) & (y[r] == j)).any()]) for k in names}
        C = {k: crit(V[k], r, maj) for k in names}; pick["oracle"][r] = V[max(names, key=lambda k: ba[k])][r]
        for c in ("conf", "balance", "agree", "fisher"):
            pick[c][r] = V[max(names, key=lambda k: C[k][c])][r]
    print(f"{bn}: lam 0.15 {ref.mean():.2f} | " + " ".join(f"{k} {G.scores(V[k], p).mean() - ref.mean():+.2f}" for k in names), flush=True)
    for c, P in pick.items():
        d = G.scores(P, p) - ref; bt = np.percentile([d[rng.integers(0, 10, 10)].mean() for _ in range(4000)], [10, 90])
        print(f"  pick by {c:8s}: {d.mean():+.2f} [{bt[0]:+.2f}, {bt[1]:+.2f}] | P04 {d[3]:+.1f} | " + " ".join(f"{v:+.1f}" for v in d), flush=True)
