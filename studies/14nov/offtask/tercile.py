"""TERCILE (ledger s241 step 1): v6f's emitted accuracy by tercile of the rest net's p_rest within each session, new people.
Chain outputs: runs/chainout/s215_<order>_off.npz, <base>|V6Fr (recording layout, all_loso rows = window index).
p_rest: seed mean of p4[:, 3] from runs/diag/restprest/<order>/rest4_sh_r1-all_loso-s*.npz (the order's own re-stream).
Accuracy per (session, tercile), session-equal mean, mean over the six bases. Gate: bottom - top >= 3.0 points (mean over orders).
Usage: python studies/14nov/offtask/tercile.py [order ...]   (default: within0 within1 full0 full1 runs0 runs1)"""
import glob, os, sys
import numpy as np
REPO = os.path.abspath(__file__ + "/../../../.."); sys.path.insert(0, REPO + "/replay")
import env  # noqa: E402
p0 = "all_loso"; Z = env.A.L.Z; y, ses, per = Z[f"y|{p0}"].astype(int), Z[f"sess|{p0}"], Z[f"person|{p0}"]; N = len(y)
ORDERS = sys.argv[1:] or ["within0", "within1", "full0", "full1", "runs0", "runs1"]


def prest(order):
    fs = sorted(glob.glob(f"{REPO}/runs/diag/restprest/{order}/rest4_sh_r1-{p0}-s*.npz")); assert len(fs) == 3, (order, fs)
    out = []
    for f in fs:
        z = np.load(f); assert (z["index"] == np.arange(N)).all(), f; out.append(z["p4"][:, 3])
    return np.mean(out, 0), np.mean([np.load(f)["proba"] for f in fs], 0)


def by_tercile(correct, pr):
    acc = np.zeros((len(dict.fromkeys(ses)), 3)); pp = []
    for i, s in enumerate(dict.fromkeys(ses)):
        r = np.flatnonzero(ses == s); k = np.array_split(r[np.argsort(pr[r], kind="stable")], 3)
        acc[i] = [correct[q].mean() for q in k]; pp.append(per[r[0]])
    return acc, np.array(pp)


GAP, GAPP = [], []
for o in ORDERS:
    pr, rp = prest(o); z = np.load(f"{REPO}/runs/chainout/s215_{o}_off.npz"); bases = sorted({k.split("|")[0] for k in z.files})
    accs = [by_tercile(z[f"{b}|V6Fr"].argmax(1) == y, pr) for b in bases]; acc = np.mean([a for a, _ in accs], 0) * 100; pp = accs[0][1]
    own, _ = by_tercile(rp.argmax(1) == y, pr); gap = acc[:, 0] - acc[:, 2]
    gp = np.array([gap[pp == q].mean() for q in np.unique(pp)]); GAP.append(gap.mean()); GAPP.append(gp)
    print(f"{o:8s} p_rest mean {pr.mean():.3f} q10/50/90 {np.percentile(pr, [10, 50, 90]).round(3)} | v6f acc by tercile (low->high p_rest) "
          f"{acc.mean(0).round(2)} gap {gap.mean():+.2f} | rest net's own acc {(own.mean(0) * 100).round(1)} | people with gap>0 {int((gp > 0).sum())}/10", flush=True)
gp = np.mean(GAPP, 0); rng = np.random.default_rng(0); bt = np.percentile([gp[rng.integers(0, 10, 10)].mean() for _ in range(4000)], [10, 90])
print(f"MEAN gap (bottom - top tercile) {np.mean(GAP):+.2f} points [80 % person {bt[0]:+.2f}, {bt[1]:+.2f}] per person " + " ".join(f"{v:+.1f}" for v in gp))
print("GATE", "PASS" if np.mean(GAP) >= 3.0 else "FAIL (top tercile not 3 points worse: stop)")
