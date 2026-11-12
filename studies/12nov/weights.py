"""WEIGHTS (ledger s66): the 6 Nov retune grid read under the dev objective J and a sealed-like objective S.
Usage: python replay/grid.py runs/grid_12nov.npz B2_pool ; python studies/12nov/weights.py runs/grid_12nov.npz"""
import sys
import numpy as np
z = np.load(sys.argv[1]); PEOPLE = [f"P{i:02d}" for i in range(1, 11)]; G1 = [0, 1, 4, 6, 8]; G2 = [2, 3, 5, 7, 9]
cfgs = sorted({k.split("|")[1] + "|" + k.split("|")[2] for k in z.files if "|pre|" not in k}); base = z.files[0].split("|")[0]
N = np.array([z[f"{base}|{c}|all_loso"] for c in cfgs]); S_ = np.array([z[f"{base}|{c}|seen_session"] for c in cfgs])  # (n_cfg, 10), (n_cfg, 2) = P04, P10
J = 0.5 * N.mean(1) + 0.5 * S_.mean(1); S = 0.566 * N.mean(1) + 0.434 * (0.87 * S_[:, 1] + 0.13 * S_[:, 0])
ship = cfgs.index("1.15,0.5,2.0,5.0|0.15"); print(f"{len(cfgs)} settings (alpha, EM lambda, slot weight, teacher weight | TSEM lambda); shipped {cfgs[ship]}: J {J[ship]:.2f} S {S[ship]:.2f} new {N[ship].mean():.2f} P04 {S_[ship, 0]:.2f} P10 {S_[ship, 1]:.2f}")
for name, obj in (("J (dev)", J), ("S (sealed-like)", S)):
    o = np.argsort(-obj)[:5]; print(f"top by {name}:")
    for i in o:
        print(f"   {cfgs[i]:26s} J {J[i]:.2f} ({J[i] - J[ship]:+.2f}) S {S[i]:.2f} ({S[i] - S[ship]:+.2f}) new {N[i].mean():.2f} P04 {S_[i, 0]:.2f} P10 {S_[i, 1]:.2f} | rank of the shipped setting {int((obj > obj[ship]).sum()) + 1}")
print(f"spread of S over the grid: min {S.min():.2f} median {np.median(S):.2f} max {S.max():.2f}; shipped at percentile {100 * (S < S[ship]).mean():.0f}")
res = []
for tr, te, nm in ((G1, G2, "choose on G1, score on G2"), (G2, G1, "choose on G2, score on G1")):
    i = np.argmax(0.566 * N[:, tr].mean(1) + 0.434 * S_[:, 1]); d = N[i, te].mean() - N[ship, te].mean(); res.append(d)
    print(f"nested, {nm}: picked {cfgs[i]}; new people of the other half {d:+.2f} (P10 {S_[i, 1] - S_[ship, 1]:+.2f}, P04 {S_[i, 0] - S_[ship, 0]:+.2f})")
print(f"nested mean on new people {np.mean(res):+.2f}")
print("per-person optimum over the grid (best setting for each new person alone, gain over shipped): " + " ".join(f"{p[1:]}:{N[:, j].max() - N[ship, j]:+.1f}" for j, p in enumerate(PEOPLE)))
