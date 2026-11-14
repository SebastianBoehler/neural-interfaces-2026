"""OFFTASK compare (ledger s241 step 2a): weighted-template arms against the unweighted v6f chain, new people, per order and
mean over orders: change of the person-equal cell balanced accuracy (mean over the six bases), 80 % person interval,
people up, without P10. Reference: runs/chainout/s215_<order>_off.npz (the W0 hook run must equal it, checked when present).
Usage: python studies/14nov/offtask/compare.py ARM [order ...]"""
import os, sys
import numpy as np
REPO = os.path.abspath(__file__ + "/../../../.."); sys.path.insert(0, REPO + "/replay")
import env  # noqa: E402
SC, p0 = env.A.SC, "all_loso"; ARM = sys.argv[1]; ORDERS = sys.argv[2:] or ["within0", "within1", "full0", "full1", "runs0", "runs1"]
rng = np.random.default_rng(0)


def ci(d):
    return np.percentile([d[rng.integers(0, len(d), len(d))].mean() for _ in range(4000)], [10, 90])


def per_person(z, arm="V6Fr"):
    bases = sorted({k.split("|")[0] for k in z.files}); return np.mean([SC.person(z[f"{b}|{arm}"].astype(np.float64), p0) for b in bases], 0)


D = []
for o in ORDERS:
    ref = np.load(f"{REPO}/runs/chainout/s215_{o}_off.npz"); w0 = f"{REPO}/runs/diag/offtask/W0_{o}.npz"
    if os.path.exists(w0):
        z0 = np.load(w0); dd = max(np.abs(z0[k] - ref[k]).max() for k in ref.files); assert dd < 1e-6, f"W0 {o} differs from s215 off by {dd:.2e}"
    z = np.load(f"{REPO}/runs/diag/offtask/{ARM}_{o}.npz"); d = per_person(z) - per_person(ref); D.append(d); b = ci(d)
    p0r = per_person(z, "P0r") - per_person(ref, "P0r")
    print(f"{o:8s} {ARM} - v6f: {d.mean():+.2f} [{b[0]:+.2f}, {b[1]:+.2f}] up {int((d > 0).sum())}/10 without P10 {np.delete(d, 9).mean():+.2f} | "
          f"same hook on v5.1b (P0r) {p0r.mean():+.2f} | per person " + " ".join(f"{v:+.1f}" for v in d), flush=True)
d = np.mean(D, 0); b = ci(d)
print(f"MEAN over {len(ORDERS)} orders {ARM} - v6f: {d.mean():+.2f} [80 % person {b[0]:+.2f}, {b[1]:+.2f}] people up {int((d > 0).sum())}/10 "
      f"without P10 {np.delete(d, 9).mean():+.2f} | per person " + " ".join(f"{v:+.2f}" for v in d))
print("KILL RULE (mean < +0.3 or people up < 7):", "PASS" if d.mean() >= 0.3 and (d > 0).sum() >= 7 else "FAIL" + (" (inside noise, interval spans 0: undecided)" if abs(d.mean()) < 0.3 and b[0] < 0 < b[1] else ""))
