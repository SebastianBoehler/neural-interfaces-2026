"""BAGSCORE (ledger s223): scores runs/bagging/bagcore.npz. Member alone (cell-mean BA per person, new = all_loso,
seen = seen_session): the full fit, the mean single bag, and the bagged averages (l1, l2, l1+l2; arithmetic and
geometric). Then the recording-order chain (A.system, B2_pool, all families) with core1 replaced by the full fit and by
each bagged average, keeping the Hilbert slot hb = 2 Ps[0] - c1 as stored. check.py must print OK first.
Usage: python studies/14nov/bagging/bagscore.py [path]"""
import os, sys
import numpy as np
REPO = os.path.abspath(__file__ + "/../../../..")
sys.path[:0] = [REPO + "/replay"]
import env  # noqa: E402
A = env.A; Z = A.L.Z; SC = A.SC
F = np.load(sys.argv[1] if len(sys.argv) > 1 else REPO + "/runs/bagging/bagcore.npz")
C1 = {p: A.C1[p].copy() for p in A.PROTOS}; IDX = {p: Z[f"idx|{p}"].astype(int) for p in A.PROTOS}


def geo(S):
    L = np.log(np.clip(S, 1e-6, 1)).mean(0); q = np.exp(L - L.max(-1, keepdims=True)); return q / q.sum(-1, keepdims=True)


arms = {}
for p in A.PROTOS:
    full = F[f"{p}|full"][IDX[p]]; l1, l2 = F[f"{p}|l1"][:, IDX[p]], F[f"{p}|l2"][:, IDX[p]]
    c = np.corrcoef(full.ravel(), C1[p].ravel())[0, 1]; print(f"{p}: full fit against the stored core1 member, correlation {c:.4f}")
    assert c >= 0.99, "the full fit does not reproduce core1"
    for name, P in (("full", full), ("l1", l1.mean(0)), ("l2", l2.mean(0)), ("l1+l2", np.concatenate([l1, l2]).mean(0)),
                    ("l1 geo", geo(l1)), ("l2 geo", geo(l2))):
        arms.setdefault(name, {})[p] = P
    arms.setdefault("bag mean", {})[p] = [SC.person(b, p).mean() for b in np.concatenate([l1, l2])]
print("member alone        new    seen  | per person new")
fn = SC.person(arms["full"]["all_loso"], "all_loso")
for name, d in arms.items():
    if name == "bag mean":
        print(f"{'single bag (mean)':18s} {np.mean(d['all_loso']):6.2f} {np.mean(d['seen_session']):6.2f}  | "
              f"range new {min(d['all_loso']):.2f}-{max(d['all_loso']):.2f}")
        continue
    n, s = SC.person(d["all_loso"], "all_loso"), SC.person(d["seen_session"], "seen_session")
    print(f"{name:18s} {n.mean():6.2f} {s.mean():6.2f}  | " + " ".join(f"{v - f:+.1f}" for v, f in zip(n, fn)))

orig = A.members


def members(bn, p):
    out = orig(bn, p); out["hb"] = 2 * Z[f"{bn}|Ps|{p}"][0] - C1[p]; return out


A.members = members
print("chain (B2_pool, recording order)   J      new    seen   dJ vs full")
j0 = None
for name in ("full", "l1", "l2", "l1+l2", "l1 geo"):
    for p in A.PROTOS:
        A.C1[p] = arms[name][p]
    s = A.summary(A.system("B2_pool", A.FAMS)); j = 0.5 * s["dn"].mean() + 0.5 * s["ds"].mean(); j0 = j if j0 is None else j0
    print(f"{name:32s} {j:7.3f} {s['dn'].mean():6.2f} {s['ds'].mean():6.2f}  {j - j0:+.2f}  per person new "
          + " ".join(f"{v:.1f}" for v in s["dn"]), flush=True)
for p in A.PROTOS:
    A.C1[p] = C1[p]
A.members = orig
print("mean seconds per LR fit", float(F["secs"].mean()), "n fits", len(F["secs"]))
