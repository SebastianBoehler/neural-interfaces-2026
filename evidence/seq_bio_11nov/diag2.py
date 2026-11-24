"""Calibration check: how sharp are chain posteriors, and what weight should the Graz position prior get?"""
import sys, collections
import numpy as np
from scipy.optimize import minimize
ARCH = "<repo>/runs/archive_scratch_11nov"
sys.path.insert(0, ARCH + "/secop")
import env
A = env.A; M = A.M; SC = A.SC
z = np.load(env.REPO + "/data/proteus_windows.npz", allow_pickle=True)
y = z["y"]; per = np.array([s[-3:] for s in z["subject"]]); ses = z["session"]; task = z["task"].astype(str); run = z["run"]
N = len(y); pos = np.zeros(N, int)
for k in set(zip(per, ses, run)):
    m = np.flatnonzero((per == k[0]) & (ses == k[1]) & (run == k[2])); pos[m] = np.arange(len(m))
graz = np.char.endswith(task, "Graz"); pm = pos % 3
T = np.full((3, 3), 0.22); T[0, 2] = T[1, 1] = T[2, 0] = 0.56
LP = np.where(graz[:, None], np.log(3 * T[pm]), 0.0)
p = "all_loso"; bn = "B2_pool"
mixv, _ = A.mix(bn, p, A.FAMS); b = M.rebal(mixv, p, None); g = A.gm.gate(b, p); e = A.gm.em_g(b, p, g); t = A.gm.tsem_g(e, p, g)
def ba_cells(pred, sel):
    cell, pp = SC.B[f"cell|{p}"], SC.B[f"person|{p}"]; acc = collections.defaultdict(list)
    for c in np.unique(cell[sel]):
        m = cell == c; acc[pp[m][0]].append(SC.cell_ba(y[m], pred[m]))
    return np.mean([np.mean(v) for v in acc.values()])
for name, P in (("mix", mixv), ("rebal", b), ("tsem", t)):
    s = np.sort(P, 1); top = s[:, -1]; ok = P.argmax(1) == y
    print(f"\n{name}: mean top prob {top.mean():.3f} (acc {ok.mean():.3f}); median top1/top2 odds {np.median(s[:, -1] / s[:, -2]):.2f}; quartiles top {np.round(np.quantile(top, [.1, .25, .5, .75, .9]), 2)}")
    lP = np.log(np.clip(P, 1e-9, 1))
    base = ba_cells(P.argmax(1), graz)
    for gg in (0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5):
        print(f"   fixed g {gg}: Graz cell BA {ba_cells((lP + gg * LP).argmax(1), graz) - base:+.2f}", end=";")
    print()
    # LOPO fit of temperature a and prior weight g on Graz rows (multinomial NLL)
    def nll(th, m):
        s_ = th[0] * lP[m] + th[1] * LP[m]; s_ = s_ - s_.max(1, keepdims=True)
        return -(s_[np.arange(m.sum()), y[m]] - np.log(np.exp(s_).sum(1))).mean()
    pred = P.argmax(1).copy(); ths = []
    for q in np.unique(per):
        tr = graz & (per != q); th = minimize(nll, [1.0, 0.5], args=(tr,), method="Nelder-Mead").x; ths.append(th)
        te = graz & (per == q); pred[te] = (th[0] * lP[te] + th[1] * LP[te]).argmax(1)
    ths = np.array(ths)
    print(f"   LOPO fit: temperature a {ths[:, 0].mean():.2f} (sd {ths[:, 0].std():.2f}), prior weight g {ths[:, 1].mean():.2f} (sd {ths[:, 1].std():.2f}); ratio g/a {np.mean(ths[:, 1] / ths[:, 0]):.3f}")
    print(f"   LOPO Graz cell BA {base:.2f} -> {ba_cells(pred, graz):.2f} ({ba_cells(pred, graz) - base:+.2f}); window acc {np.mean(P.argmax(1)[graz] == y[graz]):.3f} -> {np.mean(pred[graz] == y[graz]):.3f}")
