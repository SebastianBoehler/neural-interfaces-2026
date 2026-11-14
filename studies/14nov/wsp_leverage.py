"""WSP LEVERAGE (ledger s207): chain points per accuracy point added to one component (shufcombo SYN_T/SYN_D runs).
For each component and order: (V6Fr with +D points on the component) - (V6Fr of the regular file), per person, mean over
the six bases; leverage = that / D. Also P0r (v5.1b's chain), to see whether the lever is v5.1b's or the v6 additions'.
Usage: python studies/14nov/wsp_leverage.py D order [order ...]"""
import glob, os, sys
import numpy as np
REPO = os.path.abspath(__file__ + "/../../.."); D = sys.argv[1]; ORDERS = sys.argv[2:]
COMP = [f"s{k}" for k in range(16)] + ["c1", "rest", "eye"]


def m(f, arm):
    z = np.load(f); b = sorted({k.split("|")[0] for k in z.files}); assert len(b) == 6, f; return np.mean([z[f"{x}|{arm}"] for x in b], 0)


print(f"WSP LEVERAGE (+{D} points on one component; chain change per point, new people, mean over bases; V6Fr | P0r)")
rows = {}
for c in COMP:
    line, got = f"  {c:5s}", []
    for o in ORDERS:
        base = f"{REPO}/runs/shufcombo_{o}_ocu_gC5_all_ge_hb_ts_gp0_ttt18{o}L.npz"; f = base[:-4] + f"_syn{c}{D}.npz"
        if not os.path.exists(f):
            line += f" | {o} -"; continue
        dv, dp = m(f, "V6Fr") - m(base, "V6Fr"), m(f, "P0r") - m(base, "P0r"); got.append(dv.mean() / float(D))
        line += f" | {o} {dv.mean() / float(D):+.3f} ({dp.mean() / float(D):+.3f}) P10 {dv[9] / float(D):+.3f}"
    rows[c] = got; print(line + (f" | mean {np.mean(got):+.3f}" if got else ""), flush=True)
print("Rule (s207): >= 0.05 per point under at least two schemes -> stage 2; below 0.03 everywhere -> no lever; else undecided")
for c, g in rows.items():
    if len(g) == len(ORDERS):
        v = "stage 2" if sum(x >= 0.05 for x in g) >= 2 else ("no lever" if max(g) < 0.03 else "undecided")
        print(f"  {c}: {v}")
