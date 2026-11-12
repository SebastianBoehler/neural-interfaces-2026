"""POPPRIOR (ledger s92): population class-template prior for TSEM (new people, grid chain, six bases).
Usage: python studies/12nov/popprior.py"""
import glob, os, sys
import numpy as np
REPO = os.path.abspath(__file__ + "/../../..")
sys.path.insert(0, REPO + "/replay")
import env  # noqa: E402
import grid as G  # noqa: E402
import pl  # noqa: E402
p = "all_loso"; Z = G.Z; sess = Z[f"sess|{p}"]; rng = np.random.default_rng(0); PRI = {}
assert (pl.B[f"sess|{p}"] == sess).all() and np.allclose(pl.pc(p), Z[f"pc128|{p}"])
for f in sorted(glob.glob(f"{REPO}/runs/replay_data/feats_exact/all_loso-P*.npz")):
    z = np.load(f, allow_pickle=True); X, yt, it = z["pc_tr"][:, :128], z["y_tr"].astype(int), z["idx_tr"].astype(int); mus = []
    for s in np.unique(sess[it]):
        m = sess[it] == s; Xs = (X[m] - X[m].mean(0)) / np.maximum(X[m].std(0), 1e-3); mus.append(np.stack([Xs[yt[m] == k].mean(0) for k in range(3)]))
    pm = np.mean(mus, 0)
    for k in [x for x in z.files if x.endswith("|idx")]:
        for s in np.unique(sess[z[k].astype(int)]):
            PRI[s] = pm
assert set(PRI) == set(np.unique(sess))
pos = np.zeros(len(sess))
for s in np.unique(sess):
    r = np.flatnonzero(sess == s); pos[r] = np.arange(len(r)) / len(r)
y = Z[f"y|{p}"].astype(int); R = {}
for bn in env.BASES:
    e = G.pre_tsem(bn, p, *G.SHIP); ref = pl.tsem(e, p); r0 = G.scores(ref, p)
    for m0p in (10.0, 30.0):
        for nmin in (15, 5):
            out = pl.tsem(e, p, priors=PRI, m0p=m0p, nmin=nmin); d = G.scores(out, p) - r0; R[bn, m0p, nmin] = d
            f = pos < 0.2; print(f"{bn} m0p {m0p} start {nmin}: new {d.mean():+.2f} | first fifth window acc {100 * np.mean(out[f].argmax(1) == y[f]) - 100 * np.mean(ref[f].argmax(1) == y[f]):+.1f} | " + " ".join(f"{x:+.1f}" for x in d), flush=True)
arms = [(m, n) for m in (10.0, 30.0) for n in (15, 5)]; best = max(arms, key=lambda a: np.mean([R[b, *a].mean() for b in env.EXPL]))
fr = np.mean([R[b, *best] for b in env.FRESH], 0); bt = np.percentile([fr[rng.integers(0, 10, 10)].mean() for _ in range(4000)], [10, 90])
print(f"chosen {best}: exploration {np.mean([R[b, *best].mean() for b in env.EXPL]):+.2f} | FRESH {fr.mean():+.2f} [{bt[0]:+.2f}, {bt[1]:+.2f}] | people >= 0 {(fr >= 0).sum()}/10")
