"""s248 M3b / M4b / M4c (amendment): where the hardware run fingerprint sits, whether runs cluster without labels, and
what the clusters are worth as groups of the run-level count filter under full shuffle.
M3b: cross-run max |r| with each channel demeaned per window; where the top pairs are.
M4b: 1-NN run identity per feature block; per-session Ward / GMM clustering (PCA 20, k = true run count): purity, ARI.
M4c: offline Ward clusters -> runcount.filt groups on full0/full1 (V6Fr, P0r; six bases, new people) vs true runs.
Writes runs/levers/hw_clusters.npz (cluster label per proteus_windows row).
Usage: python studies/14nov/levers/fingerprint_cluster.py"""
import os, sys
import numpy as np
from scipy.signal import butter, sosfiltfilt
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score
from sklearn.mixture import GaussianMixture
REPO = os.path.abspath(__file__ + "/../../../..")
with np.load(REPO + "/data/proteus_windows.npz") as w:
    X, sub, ses, run, task = w["X"].astype(np.float32), w["subject"], w["session"], w["run"], w["task"]
SID = np.char.add(sub.astype(str), np.char.add("|", ses.astype(str))); RID = np.char.add(SID, np.char.add("|", run.astype(str)))
FS = 120.0

F = X - X.mean(-1, keepdims=True); F = F.reshape(len(X), -1).astype(np.float64); F /= np.linalg.norm(F, axis=1, keepdims=True)
C = np.abs(F @ F.T); np.fill_diagonal(C, 0); same = RID[:, None] == RID[None, :]; C[same] = 0
i, j = np.unravel_index(np.argsort(C, axis=None)[::-1][:10], C.shape)
print(f"M3b channel-demeaned: max |r| across runs {C.max():.3f}; pairs > 0.9 {(C > 0.9).sum() // 2}, > 0.7 {(C > 0.7).sum() // 2}")
for a, b in list(zip(i, j))[::2]:
    print(f"   {C[a, b]:.3f}  {RID[a]} {task[a]}  <->  {RID[b]} {task[b]}")
del F, C


def bp(lo, hi):
    return sosfiltfilt(butter(4, [lo / (FS / 2), min(hi / (FS / 2), 0.99)], btype="band", output="sos"), X, axis=-1)


hb = bp(30, 58); iu = np.triu_indices(41, 1)
blocks = {"hp 45-58": np.log(bp(45, 58).var(-1) + 1e-6), "line 48-52": np.log(bp(48, 52).var(-1) + 1e-6),
          "corr 30-58": np.stack([np.corrcoef(h)[iu] for h in hb])}
blocks["all hardware"] = np.concatenate(list(blocks.values()), 1)
z = lambda A: (A - A.mean(0)) / (A.std(0) + 1e-9)  # noqa: E731


def nn1(Z, lab):
    D = ((Z[:, None] - Z[None]) ** 2).sum(-1); np.fill_diagonal(D, np.inf); return (lab[D.argmin(1)] == lab).mean()


def purity(lab, cl):
    return sum(np.unique(lab[cl == c], return_counts=True)[1].max() for c in np.unique(cl)) / len(lab)


CL = np.empty(len(X), dtype=object)
for name, Fe in blocks.items():
    r1, pw, aw, pg, ag = [], [], [], [], []
    for s in np.unique(SID):
        m = SID == s; lab = RID[m]; k = len(np.unique(lab)); Z = z(Fe[m]); r1.append(nn1(Z, lab))
        P = PCA(min(20, Z.shape[1]), random_state=0).fit_transform(Z)
        cw = AgglomerativeClustering(k, linkage="ward").fit_predict(P); pw.append(purity(lab, cw)); aw.append(adjusted_rand_score(lab, cw))
        cg = GaussianMixture(k, covariance_type="diag", random_state=0, n_init=3).fit(P).predict(P); pg.append(purity(lab, cg)); ag.append(adjusted_rand_score(lab, cg))
        if name == "all hardware":
            CL[np.flatnonzero(m)] = [f"hw{c}" for c in cw]
    print(f"M4b {name:13s}: 1-NN {np.mean(r1):.3f} | Ward purity {np.mean(pw):.3f} ARI {np.mean(aw):.3f} | GMM purity {np.mean(pg):.3f} ARI {np.mean(ag):.3f}")
os.makedirs(REPO + "/runs/levers", exist_ok=True); np.savez(REPO + "/runs/levers/hw_clusters.npz", cl=CL.astype(str))

sys.path.insert(0, REPO + "/studies/14nov"); _argv, sys.argv = sys.argv, sys.argv[:1]
import runcount as R  # noqa: E402
sys.argv = _argv
hwlab = CL.astype(str)[R.IDX]; G = {u: np.flatnonzero(np.char.add(R.ses.astype(str), hwlab) == u) for u in dict.fromkeys(np.char.add(R.ses.astype(str), hwlab))}
print("M4c count filter groups under full shuffle (change vs raw, new people, six bases): true runs | hardware Ward clusters (cap 12)")
for order in ("full0", "full1"):
    rk = R.rank(order); zz = np.load(f"{REPO}/runs/chainout/s215_{order}_off.npz"); bases = sorted({k.split("|")[0] for k in zz.files})
    for a in ("P0r", "V6Fr"):
        dt, dh = [], []
        for bn in bases:
            P = zz[f"{bn}|{a}"].astype(np.float64); b0 = R.SC.person(P, R.p0)
            dt.append(R.SC.person(R.filt(P, rk), R.p0) - b0); dh.append(R.SC.person(R.filt(P, rk, G), R.p0) - b0)
        dt, dh = np.mean(dt, 0), np.mean(dh, 0)
        print(f"   {order} {a:4s} true {dt.mean():+.2f} | hw {dh.mean():+.2f} ({100 * dh.mean() / dt.mean():.0f} %; people up {(dh > 0).sum()}/10, min {dh.min():+.2f})", flush=True)
