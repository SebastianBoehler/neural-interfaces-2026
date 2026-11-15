"""s270 collect: python collect.py RUNDIR_SUFFIX_DIR (e.g. runs/dreyer270). Z41 is checked first (a0 on 41 channels within
0.001 point of core1's stored all_loso lr10), then the D27 arms are read against a0. Fails loudly on a missing fold or key mismatch."""
import json, os, sys
import numpy as np
D = sys.argv[1]; HERE = os.path.dirname(os.path.abspath(__file__))
REF = json.load(open(HERE + "/../../../runs/diag/ts_core/res/car_ov17_fs2_t0.1_le_k2.json"))["heads"]["lr10"]["all_loso"]

def folds(v):
    fs = [json.load(open(f"{D}/{v}/fold{i}.json")) for i in range(10)]
    assert len({f["key"] for f in fs}) == 1, f"{v}: mixed cache keys"
    return {f["person"]: f for f in fs}

z = folds("Z41"); worst = max(abs(z[p]["ba"]["a0"] - REF[p]) for p in REF)
print(f"Z41 vs core1: max |diff| {100 * worst:.4f} points -> {'PASS' if worst < 1e-5 else 'FAIL'}")
f = folds("D27"); P = sorted(REF); arms = list(f[P[0]]["ba"])
assert all(f[p]["arms"][a]["dr"] == (a != "a0") for p in P for a in arms) and all(f[p]["n_train"]["a0"][1] == 0 for p in P)
b = {a: np.array([f[p]["ba"][a] for p in P]) for a in arms}
for a in arms:
    d = b[a] - b["a0"]
    print(f"{a:7s} {100 * b[a].mean():.2f}  vs a0 {100 * d.mean():+.2f}  up {(d > 0).sum()}/10  P04 {100 * d[P.index('P04')]:+.1f} P10 {100 * d[P.index('P10')]:+.1f}  per person",
          " ".join(f"{100 * x:+.1f}" for x in d), "| MI share of predictions", f"{np.mean([f[p]['pred_share'][a][0] / sum(f[p]['pred_share'][a]) for p in P]):.3f}")
l1 = (b["a1"] - b["a0"]).mean(); l2 = (b["a2w1"] - b["a0"]).mean(); u2 = int(((b["a2w1"] - b["a0"]) > 0).sum())
print(f"s270 readout: arm1 loss {100 * l1:+.2f} (pred <= -3, people up <= 2: up {(b['a1'] > b['a0']).sum()}); arm2 w1 removes {100 * (1 - l2 / l1) if l1 < 0 else float('nan'):.0f}% of it (pred >= 60%); "
      f"rule on arm2 w1: delta {100 * l2:+.2f}, up {u2}/10 -> {'pre-register net form' if l2 >= .01 and u2 >= 7 else 'closed in MI-label form' if l2 < -.005 else 'undecided'}")
