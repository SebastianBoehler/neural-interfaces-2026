"""WSP STAGE 2 summary (ledger s207): chain change of each final-layer edit (shufcombo OUTX runs) against the unedited
file, new people, per order (V6Fr = the v6f combination; P0r = v5.1b's chain), with the edit's held-out effect on the
component itself (from runs/s207b_fit_<comp>_<fam>.log): accuracy points gained = 'reachable points'.
Usage: python studies/14nov/wsp_stage2.py order [order ...]"""
import glob, os, re, sys
import numpy as np
REPO = os.path.abspath(__file__ + "/../../.."); ORDERS = sys.argv[1:]


def m(f, arm):
    z = np.load(f); b = sorted({k.split("|")[0] for k in z.files}); return np.mean([z[f"{x}|{arm}"] for x in b], 0)


print("WSP STAGE 2: chain change of each final-layer edit (new people, mean over bases); component accuracy held out")
for f in sorted(glob.glob(f"{REPO}/runs/wsp/final_*.npz")):
    tag = os.path.basename(f)[6:-4]; comp, fam = tag.split("_"); log = f"{REPO}/runs/s207b_fit_{tag}.log"
    acc = [(float(a), float(b)) for a, b in re.findall(r"acc ([\d.]+) -> ([\d.]+)", open(log).read())] if os.path.exists(log) else []
    reach = 100 * np.mean([b - a for a, b in acc]) if acc else np.nan; line, got = f"  {tag:14s} reachable {reach:+.2f} pts", []
    for o in ORDERS:
        base = f"{REPO}/runs/shufcombo_{o}_ocu_gC5_all_ge_hb_ts_gp0_ttt18{o}L.npz"; g = base[:-4] + f"_ox{tag}.npz"
        if not os.path.exists(g):
            line += f" | {o} -"; continue
        dv, dp = m(g, "V6Fr") - m(base, "V6Fr"), m(g, "P0r") - m(base, "P0r"); got.append(dv.mean())
        line += f" | {o} {dv.mean():+.2f} ({dp.mean():+.2f}) P10 {dv[9]:+.2f} up {int((dv > 0).sum())}/10"
    print(line + (f" | mean {np.mean(got):+.2f}" if got else ""), flush=True)
