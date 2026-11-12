"""FOCAL (ledger s87): the four SessionBN families replaced by their focal-loss versions, against the same recipe trained
through the same class (twin), routed chain, six bases. Usage: python studies/12nov/focal_swap.py"""
import glob, os, sys
import numpy as np
REPO = os.path.abspath(__file__ + "/../../..")
sys.path.insert(0, REPO + "/replay")
import env  # noqa: E402
A = env.A; SC = A.SC; D = REPO + "/runs/netaug/probs"; SLOT = {"fbraw": 5, "fb": 6, "sinc": 7, "fbl": 8}; rng = np.random.default_rng(0)
IDX = {p: A.pl.B[f"idx|{p}"].astype(int) for p in A.PROTOS}
for a in SLOT:
    for v in ("twin", "focal1", "focal2"):
        A.EXTRA[f"{a}_{v}"] = {}
        for p in A.PROTOS:
            zs = [np.load(f) for f in sorted(glob.glob(f"{D}/na_{a}_{v}-{p}-s*.npz"))]; assert len(zs) == 3 and all((z["index"] == IDX[p]).all() for z in zs), (a, v, p)
            A.EXTRA[f"{a}_{v}"][p] = np.mean([z["proba"] for z in zs], 0)


def swap(v):
    fams, extra = list(A.FAMS), []
    for a, slot in SLOT.items():
        fam = next(f for f, items in A.FAM.items() if any(k == slot for k, _ in items))
        if fam in fams:
            fams.remove(fam); extra += [(k, w) for k, w in A.FAM[fam]]
        extra = [(k, w) for k, w in extra if k != slot] + [(f"{a}_{v}", 1.0)]
    return fams, extra


R = {}
for bn in env.BASES:
    s = {v: A.summary(A.system(bn, *swap(v))) for v in ("twin", "focal1", "focal2")}
    for v in ("focal1", "focal2"):
        R[bn, v] = (s[v]["dn"] - s["twin"]["dn"], s[v]["ds"] - s["twin"]["ds"])
    print(f"{bn}: " + " | ".join(f"{v} against twin: new {R[bn, v][0].mean():+.2f} seen {R[bn, v][1].mean():+.2f}" for v in ("focal1", "focal2")), flush=True)
for v in ("focal1", "focal2"):
    d = np.mean([R[b, v][0] for b in env.BASES], 0); bt = np.percentile([d[rng.integers(0, 10, 10)].mean() for _ in range(4000)], [10, 90])
    print(f"MEAN {v}: new {d.mean():+.2f} [{bt[0]:+.2f}, {bt[1]:+.2f}] people >= 0 {(d >= 0).sum()}/10 | seen {np.mean([R[b, v][1].mean() for b in env.BASES]):+.2f}")
