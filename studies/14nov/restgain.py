"""RESTGAIN (ledger s207, in-net edit of the rest net): per-filter gains on the rest net's final layer, z = sum_f g_f z_f + b,
from restblocks.py's filter contributions. Per seed (filters of different seeds are different nets) and per held-out
person, the 40 gains are fitted on the other nine people's recording-order contributions (objective acc: log loss +
REG ||g - 1||^2; corr: + LAM x (correlation of the true-class residual with the other members' + within-run lag-1
autocorrelation)). Writes the edited member, and the unedited one (tag base, gains 1) of the same re-stream, in restshuf's
layout for shufcombo's RS_DIR: runs/wsp/restedit/<tag>/<order>/rest4_sh_r1-all_loso-s<seed>.npz (proba, index).
Usage: [REG=0.1] [LAM=0.1] python studies/14nov/restgain.py acc|corr ORDER [ORDER ...]   (runs/restblocks/<order>/{rec,dlv})"""
import glob, os, sys
import numpy as np
from scipy.optimize import minimize
REPO = os.path.abspath(__file__ + "/../../.."); sys.path.insert(0, REPO + "/replay")
import env  # noqa: E402
A = env.A; p0 = "all_loso"; Z = A.L.Z; FAM, ORDERS = sys.argv[1], sys.argv[2:]
REG, LAM = float(os.environ.get("REG", 0.1)), float(os.environ.get("LAM", 0.1)); assert FAM in ("acc", "corr")
IDX = A.pl.B[f"idx|{p0}"].astype(int); y, per = Z[f"y|{p0}"].astype(int), Z[f"person|{p0}"]; PEOPLE = list(np.unique(per))
with np.load(REPO + "/data/proteus_windows.npz") as w:
    RUN = np.char.add(w["task"].astype(str), np.char.add("|", w["run"].astype(str)))[IDX]
RUN = np.char.add(Z[f"sess|{p0}"].astype(str), np.char.add("|", RUN))
PS = Z[f"B2_pool|Ps|{p0}"]; RM = np.mean([np.log(np.clip(PS[k], 1e-6, 1))[np.arange(len(y)), y] for k in range(PS.shape[0])], 0)


def load(path):
    z = np.load(path); assert (z["index"].astype(int) == IDX).all(), path; assert "contrib" in z.files, path
    return z["contrib"].astype(np.float64), z["bias"], z["proba"]


def logp(g, zf, zb):
    z = np.einsum("nfc,f->nc", zf, g) + zb; z = z - z.max(1, keepdims=True); return z - np.log(np.exp(z).sum(1, keepdims=True))


def resid(lp, r):
    v = lp[np.arange(len(r)), y[r]].copy()
    for q in np.unique(per[r]):
        m = per[r] == q; v[m] -= v[m].mean()
    return v


def loss(g, zf, zb, r):
    lp = logp(g, zf[r], zb[r]); L = -lp[np.arange(len(r)), y[r]].mean() + REG * ((g - 1) ** 2).sum()
    if FAM == "corr":
        e, m = resid(lp, r), RM[r] - RM[r].mean(); ac = [np.corrcoef(s[:-1], s[1:])[0, 1] for s in (e[RUN[r] == u] for u in np.unique(RUN[r])) if len(s) > 3]
        L += LAM * (np.corrcoef(e, m)[0, 1] + np.nanmean(ac))
    return L


tag = f"{FAM}{REG:g}" + (f"l{LAM:g}" if FAM == "corr" else ""); print(f"RESTGAIN {tag}: per seed, 40 filter gains per held-out person, fitted on the other nine")
for o in ORDERS:
    for f in sorted(glob.glob(f"{REPO}/runs/restblocks/{o}/rec/rest4_sh_r1-{p0}-s*.npz")):
        sd = f.rsplit("-s", 1)[1][:-4]; zr, br, pr = load(f); zd, bd, pd = load(f.replace("/rec/", "/dlv/")); F = zr.shape[1]
        assert np.abs(np.exp(logp(np.ones(F), zd, bd)) - pd).max() < 1e-4, f"{o} s{sd}: contributions do not reproduce the stored posteriors"
        G = np.ones((len(y), F)); acc = []
        for q in PEOPLE:
            tr, te = np.flatnonzero(per != q), np.flatnonzero(per == q)
            g = minimize(loss, np.ones(F), args=(zr, br, tr), method="L-BFGS-B", options={"maxiter": 300}).x; G[te] = g
            acc.append((np.mean(logp(np.ones(F), zr[te], br[te]).argmax(1) == y[te]), np.mean(logp(g, zr[te], br[te]).argmax(1) == y[te])))
        ed = np.exp(np.stack([logp(G[i], zd[i:i + 1], bd[i:i + 1])[0] for i in range(len(y))]))
        for t, P in ((tag, ed), ("base", pd)):
            out = f"{REPO}/runs/wsp/restedit/{t}/{o}"; os.makedirs(out, exist_ok=True); np.savez(f"{out}/rest4_sh_r1-{p0}-s{sd}.npz", proba=P, index=IDX)
        a = np.array(acc); print(f"  {o} s{sd}: held-out rest-net accuracy {100 * a[:, 0].mean():.2f} -> {100 * a[:, 1].mean():.2f} ({100 * (a[:, 1] - a[:, 0]).mean():+.2f} points)", flush=True)
print("written runs/wsp/restedit/{" + tag + ",base}/<order>")
