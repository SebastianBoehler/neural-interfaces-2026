"""COREGAIN (ledger s207, in-net edit of core1): per-block gains on core1's head, z = sum_k g_k z_k + b, from the block
logits of coreblocks.py. For each held-out person the gains are fitted on the other nine people's recording-order blocks
(objective acc: log loss + REG ||g - 1||^2; corr: + LAM x (correlation of the true-class residual with the other members'
+ its lag-1 autocorrelation within runs)). The edited member is written in recording and delivery order as a coremember
file for shufcombo (CM_NPZ): rec|all_loso is the unedited stream, so shufcombo's DC1 = edited delivery - unedited rec
turns the stored member into the edited one in delivery order.
Usage: [REG=0.1] [LAM=1] python studies/14nov/coregain.py acc|corr ORDER [ORDER ...]   (runs/coreblocks_<order>.npz each)"""
import os, sys
import numpy as np
from scipy.optimize import minimize
REPO = os.path.abspath(__file__ + "/../../.."); sys.path.insert(0, REPO + "/replay")
import env  # noqa: E402
A = env.A; p0 = "all_loso"; Z = A.L.Z; FAM, ORDERS = sys.argv[1], sys.argv[2:]
REG, LAM = float(os.environ.get("REG", 0.1)), float(os.environ.get("LAM", 1)); assert FAM in ("acc", "corr")
y, per = Z[f"y|{p0}"].astype(int), Z[f"person|{p0}"]; PEOPLE = list(np.unique(per)); IDX = A.pl.B[f"idx|{p0}"].astype(int)
with np.load(REPO + "/data/proteus_windows.npz") as w:
    RUN = np.char.add(w["task"].astype(str), np.char.add("|", w["run"].astype(str)))[IDX]
RUN = np.char.add(Z[f"sess|{p0}"].astype(str), np.char.add("|", RUN))
PS = Z[f"B2_pool|Ps|{p0}"]; RM = np.mean([np.log(np.clip(PS[k], 1e-6, 1))[np.arange(len(y)), y] for k in range(1, PS.shape[0])], 0)
CB = {o: np.load(f"{REPO}/runs/coreblocks_{o}.npz") for o in ORDERS}; B0 = CB[ORDERS[0]]
ZB, ZI = B0[f"rec|{p0}"].astype(np.float64), B0[f"rec|{p0}|b"]; K = ZB.shape[1]
for o in ORDERS:  # every file must carry the same recording-order stream
    assert np.allclose(CB[o][f"rec|{p0}|p"], B0[f"rec|{p0}|p"], atol=1e-6), f"{o}: recording-order stream differs"


def logp(g, zb, zi):
    z = np.einsum("nkc,k->nc", zb, g) + zi; z = z - z.max(1, keepdims=True); return z - np.log(np.exp(z).sum(1, keepdims=True))


def resid(lp, r):
    v = lp[np.arange(len(r)), y[r]].copy()
    for q in np.unique(per[r]):
        m = per[r] == q; v[m] -= v[m].mean()
    return v


def corr_ac(lp, r):
    e = resid(lp, r); m = RM[r] - RM[r].mean(); ac = []
    for g in np.unique(RUN[r]):
        s = e[RUN[r] == g]
        if len(s) > 3:
            ac.append(np.corrcoef(s[:-1], s[1:])[0, 1])
    return np.corrcoef(e, m)[0, 1], np.nanmean(ac)


def loss(g, r):
    lp = logp(g, ZB[r], ZI[r]); L = -lp[np.arange(len(r)), y[r]].mean() + REG * ((g - 1) ** 2).sum()
    if FAM == "corr":
        c, a = corr_ac(lp, r); L += LAM * (c + a)
    return L


G = {}
print(f"COREGAIN {FAM} REG {REG} LAM {LAM}: {K} block gains per held-out person, fitted on the other nine")
for q in PEOPLE:
    tr, te = np.flatnonzero(per != q), np.flatnonzero(per == q)
    g = minimize(loss, np.ones(K), args=(tr,), method="L-BFGS-B", options={"maxiter": 500}).x; G[q] = g
    a0 = np.mean(logp(np.ones(K), ZB[te], ZI[te]).argmax(1) == y[te]); a1 = np.mean(logp(g, ZB[te], ZI[te]).argmax(1) == y[te])
    print(f"  {q}: acc {a0:.4f} -> {a1:.4f} | gains min {g.min():.2f} max {g.max():.2f} | slow block {g[-1]:.2f}", flush=True)
GR = np.array([G[q] for q in per])  # per row, its person's gains
OUT = {f"rec|{p0}": B0[f"rec|{p0}|p"]}
for o in ORDERS:
    OUT[f"{o}|{p0}"] = np.exp(np.stack([logp(GR[i], CB[o][f"{o}|{p0}"][i:i + 1].astype(np.float64), CB[o][f"{o}|{p0}|b"][i:i + 1])[0] for i in range(len(y))]))
tag = f"{FAM}{REG:g}" + (f"l{LAM:g}" if FAM == "corr" else ""); name = f"{REPO}/runs/wsp/coregain_{tag}_{'_'.join(ORDERS)}.npz"
os.makedirs(f"{REPO}/runs/wsp", exist_ok=True); np.savez(name, gains=np.array([G[q] for q in PEOPLE]), people=np.array(PEOPLE), **OUT); print("written", name)
