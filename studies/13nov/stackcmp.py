"""STACKCMP (ledger s198): the stack of tonight's small positives against v6f_gc5, per order, from shufcombo outputs.
Stack run (ttt18 FORMS=..., labels K: the v6 chain for new people) against the package run (labels L, gC5):
  labels part  = gC5 under K - gC5 under L          form part = <form> under K - gC5 under K
  stack        = <form> under K - gC5 under L        (combination V6Fr, new people, six-base mean per person)
Usage: python studies/13nov/stackcmp.py FORM within0 within1 ...   (needs runs/shufcombo_<o>_ocu_<form>_*ttt18<o>KF.npz,
       ..._gC5_*ttt18<o>KF.npz and ..._gC5_*_gp0_ttt18<o>L.npz)"""
import glob, os, sys
import numpy as np
REPO = os.path.abspath(__file__ + "/../../.."); FORM, ORDERS = sys.argv[1], sys.argv[2:]; rng = np.random.default_rng(0)


def one(pat):
    f = sorted(glob.glob(f"{REPO}/runs/{pat}")); assert len(f) == 1, (pat, f); z = np.load(f[0]); b = sorted({k.split("|")[0] for k in z.files})
    return np.mean([z[f"{bn}|V6Fr"] for bn in b], 0)


rows = {"labels": [], "form": [], "stack": []}
for o in ORDERS:
    sfx = "_all_ge_hb_ts_gp0_"  # the most complete form only (older local variants of the same baseline must not match)
    L, KG, KF = one(f"shufcombo_{o}_ocu_gC5{sfx}ttt18{o}L.npz"), one(f"shufcombo_{o}_ocu_gC5{sfx}ttt18{o}KF.npz"), one(f"shufcombo_{o}_ocu_{FORM}{sfx}ttt18{o}KF.npz")
    rows["labels"].append(KG - L); rows["form"].append(KF - KG); rows["stack"].append(KF - L)
    print(f"  {o}: stack {np.mean(KF - L):+.2f} (labels {np.mean(KG - L):+.2f}, form {np.mean(KF - KG):+.2f})", flush=True)
for k, v in rows.items():
    v = np.array(v); d = v.mean(0); bt = np.percentile([d[rng.integers(0, 10, 10)].mean() for _ in range(4000)], [10, 90])
    print(f"STACKCMP {FORM} {k:6s} over {len(ORDERS)} orders: new {d.mean():+.2f} [{bt[0]:+.2f}, {bt[1]:+.2f}] without P10 {np.delete(d, 9).mean():+.2f} | "
          f"orders positive {int((v.mean(1) > 0).sum())}/{len(v)} | people positive {int((d > 0).sum())}/10 | per person " + " ".join(f"{x:+.1f}" for x in d), flush=True)
