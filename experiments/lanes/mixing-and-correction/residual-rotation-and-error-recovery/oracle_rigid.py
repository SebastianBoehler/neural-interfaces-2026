"""Lane Y idea #2 oracle: label-informed best rigid map (rotation/reflection + shift) of the 2-D simplex readout per cell, on stored chain outputs.
python laneY/oracle_rigid.py OUT.json   (reads work/laneZ/{full0,within0}_prof.npz and data/proteus_windows.npz; CPU, seconds)
Fits: bias (shift only), rot (O(2)+shift) on the centered log-probs; 'ins' = in-sample (upper bound), 'cf' = 2-fold cross-fit within the cell (honest)."""
import json, sys
from pathlib import Path
import numpy as np
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parents[1]
H = np.array([[1, -1, 0], [1, 1, -2]], float); H /= np.linalg.norm(H, axis=1, keepdims=True)  # 3 -> 2 plane basis


def plane(P):
    L = np.log(np.clip(P, 1e-6, 1)); return L @ H.T


def ce(z, y, W, b, reg):
    s = (z @ W.T + b) @ H; s = s - s.max(1, keepdims=True); lp = s - np.log(np.exp(s).sum(1, keepdims=True))
    return -lp[np.arange(len(y)), y].mean() + reg * np.sum(b ** 2)


def rot(th, flip):
    c, s = np.cos(th), np.sin(th); R = np.array([[c, -s], [s, c]]); return R @ np.diag([1, -1]) if flip else R


def fit(z, y, kind, reg=1e-2):
    best = (1e9, np.eye(2), np.zeros(2))
    cands = [(0.0, 0)] if kind == "bias" else [(t, f) for f in (0, 1) for t in np.linspace(0, 2 * np.pi, 24, endpoint=False)]
    for th, f in cands:
        if kind == "bias":
            r = minimize(lambda b: ce(z, y, np.eye(2), b, reg), np.zeros(2), method="L-BFGS-B"); R, b = np.eye(2), r.x; v = r.fun
        else:
            R0 = rot(th, f)
            r = minimize(lambda b: ce(z, y, R0, b, reg), np.zeros(2), method="L-BFGS-B"); R, b = R0, r.x; v = r.fun
        if v < best[0]: best = (v, R, b)
    return best[1], best[2]


def apply(z, R, b): return (z @ R.T + b) @ H  # back to 3 scores


def cell_ba(y, g):
    return 100 * np.mean([(g[y == c] == c).mean() for c in range(3) if (y == c).any()])


def main(dest):
    z = np.load(ROOT / "data/proteus_windows.npz", allow_pickle=True)
    y0, subj, sess, iface = z["y"], z["subject"], z["session"], z["interface"]
    cells = sorted({(a, b, c) for a, b, c in zip(subj, sess, iface)})
    out = {}
    rng = np.random.default_rng(0)
    for order in ("full0", "within0"):
        d = np.load(ROOT / f"work/laneZ/{order}_prof.npz")
        idx = d["index"]; y = d["y"]; assert (y == y0[idx]).all() if len(idx) == len(y0) else True
        sub, ses, itf = subj[idx], sess[idx], iface[idx]
        for sysn in ("stock", "hardcb"):
            for stage in ("mix", "tsem", "terminal"):
                P = d[f"{sysn}|base|{stage}"]; Z = plane(P)
                res = {k: [] for k in ("base", "bias_cf", "rot_cf", "bias_ins", "rot_ins")}; who = []
                for c in cells:
                    m = (sub == c[0]) & (ses == c[1]) & (itf == c[2])
                    if not m.any(): continue
                    r = np.flatnonzero(m); yy, zz = y[r], Z[r]
                    res["base"].append(cell_ba(yy, P[r].argmax(1))); who.append("|".join(map(str, c)))
                    for kind in ("bias", "rot"):
                        R, b = fit(zz, yy, kind); res[f"{kind}_ins"].append(cell_ba(yy, apply(zz, R, b).argmax(1)))
                        perm = rng.permutation(len(r)); halves = (perm[: len(r) // 2], perm[len(r) // 2:]); g = np.zeros(len(r), int)
                        for a_, b_ in (halves, halves[::-1]):
                            R, b = fit(zz[a_], yy[a_], kind); g[b_] = apply(zz[b_], R, b).argmax(1)
                        res[f"{kind}_cf"].append(cell_ba(yy, g))
                out[f"{order}|{sysn}|{stage}"] = dict(mean={k: float(np.mean(v)) for k, v in res.items()}, cells=who, per_cell=res)
                print(order, sysn, stage, {k: round(float(np.mean(v)), 2) for k, v in res.items()}, flush=True)
    json.dump(out, open(dest, "w"))


if __name__ == "__main__":
    main(sys.argv[1])
