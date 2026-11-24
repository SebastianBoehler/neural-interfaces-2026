"""Lane R4b P04/P10 from stored Lane A tapes (read only copies in work/r4b_tapes/out_<ARM>/<order>_<who>_<day>/): proxies from the HARDCB tape's own output (conf = max posterior,
ent = 1 - H/log 3; `agree` needs the personal head and is not on the tapes), gate rule of laneR4b/build_dose.py (open iff >= 10 post-recognition windows and mean of the last 40 < theta),
gated label = ARM tape label while open, hardcb tape label otherwise (post-hoc switch: the arm's adaptation state is the one it had on its own, an approximation).
usage: posthoc.py TAPE_DIR ARM THETA_CONF[,..] THETA_ENT[,..]   BA = mean recall over the 3 classes (Lane A scorer), gold from data/proteus_windows.npz by the stored row index."""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
Y = np.load(ROOT / "data/proteus_windows.npz", allow_pickle=True)["y"]


def ba(y, p):
    return float(np.mean([(p[y == c] == c).mean() for c in range(3)]))


def proxy(P, kind):
    q = P.max(1) if kind == "conf" else 1 + (P * np.log(np.clip(P, 1e-12, 1))).sum(1) / np.log(3)
    return q


def gate_open(P, ramp, kind, theta, W=40, MIN=10):
    q, post = proxy(P, kind), ramp > 0
    rec = np.flatnonzero(post)
    out = np.zeros(len(P), bool)
    if not len(rec):
        return out
    r0 = rec[0]
    for t in range(r0, len(P)):  # window t decided from post-recognition windows before t (as in the package: appended after the step)
        h = q[r0:t]
        out[t] = len(h) >= MIN and h[-W:].mean() < theta
    return out


def cells(tape_dir, arm):
    d = Path(tape_dir) / f"out_{arm}"
    for f in sorted(d.glob("*/*.npz")):
        if f.name.endswith("smoke.npz"):
            continue
        order, who, day = f.stem.split("_")
        yield order, who, day, f


def run(tape_dir, arm, th_conf, th_ent):
    res = {}
    for order, who, day, f in cells(tape_dir, arm):
        z = np.load(f)
        y = Y[z["rows"]]
        P, base, lab = z["hardcb|scores"], z["hardcb|labels"], z[f"{arm}|labels"]
        ramp = np.maximum(z["hardcb|contrast"][:, 2], z["hardcb|identity"][:, 2])
        r = {"hardcb": ba(y, base), arm: ba(y, lab), "rec": int(np.flatnonzero(ramp > 0)[0]) if (ramp > 0).any() else -1,
             "conf": float(proxy(P, "conf")[ramp > 0].mean()) if (ramp > 0).any() else float("nan"),
             "ent": float(proxy(P, "ent")[ramp > 0].mean()) if (ramp > 0).any() else float("nan")}
        for kind, ths in (("conf", th_conf), ("ent", th_ent)):
            for th in ths:
                g = gate_open(P, ramp, kind, th)
                r[f"G{kind}_{th}"] = ba(y, np.where(g, lab, base)); r[f"open_{kind}_{th}"] = float(g.mean())
        res[(order, who, int(day))] = r
    return res


if __name__ == "__main__":
    tape_dir, arm = sys.argv[1], sys.argv[2]
    tc = [float(v) for v in sys.argv[3].split(",")]; te = [float(v) for v in sys.argv[4].split(",")]
    res = run(tape_dir, arm, tc, te)
    keys = [k for k in next(iter(res.values())) if k.startswith("G") or k == arm]
    for who in ("P04", "P10"):
        rows = [r for k, r in res.items() if k[1] == who]
        print(who, "cells", len(rows), "conf mean %.3f ent mean %.3f" % (np.nanmean([r["conf"] for r in rows]), np.nanmean([r["ent"] for r in rows])),
              " ".join(f"{k} {100 * np.mean([r[k] - r['hardcb'] for r in rows]):+.2f}" for k in keys),
              " open " + " ".join(f"{k[5:]} {np.mean([r[k] for r in rows]):.2f}" for k in rows[0] if k.startswith("open_")))
