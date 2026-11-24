"""Lane D3 NETBCI scoring (local). usage: score.py D3_OUT_DIR B10_OUT_DIR DEST.json
BA2 as lane B10 (MI recall = right_hand predicted 0, specificity = rest predicted 1 or 2). Deltas paired per person over (day, order), people-resampled 80 % (10 000 draws).
R4 and the B10 hardcbB are read from the finished B10 run on the SAME two orders (full:72000, full:72001) for the matched R4 reference."""
import json
import sys
from pathlib import Path

import numpy as np

out, b10, dest = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
ORD = ("full|72000", "full|72001")
rng = np.random.default_rng(2020)


def ba2(y, lab):
    return 50 * ((lab[y == 1] == 0).mean() + (lab[y == 0] != 0).mean())


def d3_arm(a):  # (person, day, order) BA2, meta rows
    A, M = np.full((19, 3, 2), np.nan), np.full((19, 3, 2, 11), np.nan)
    for k in range(19):
        f = out / f"k{k}" / "d3.npz"
        if not f.exists():
            continue
        with np.load(f) as z:
            for o, s in enumerate(ORD):
                for d in (2, 3, 4):
                    y, lab = z[f"{s}|{d}|y"], z[f"{a}|{s}|{d}|labels"]
                    A[k, d - 2, o] = ba2(y, lab)
                    M[k, d - 2, o] = z[f"{a}|{s}|{d}|meta"]
    return A, M


def b10_arm(v):
    A = np.full((19, 3, 2), np.nan)
    for k in range(19):
        f = b10 / f"{v}_k{k}" / "ret.npz"
        if f.exists():
            with np.load(f) as z:
                for o, s in enumerate(ORD):
                    for d in (2, 3, 4):
                        A[k, d - 2, o] = ba2(z[f"{s.replace('|', '|')}|{d}|y"], z[f"{s}|{d}|labels"])
    return A


def boot(x):
    x = x[~np.isnan(x)]
    b = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(10000)]
    return [float(x.mean()), float(np.percentile(b, 10)), float(np.percentile(b, 90)), int(len(x))]


arms = {a: d3_arm(a) for a in ("hardcbB", "D3f", "D3g")}
res = {"abs": {a: float(np.nanmean(v[0])) for a, v in arms.items()}}
per = {a: np.nanmean(v[0], axis=(1, 2)) for a, v in arms.items()}  # per person
res["delta_vs_hardcbB"] = {a: boot(per[a] - per["hardcbB"]) for a in ("D3f", "D3g")}
res["gated_minus_fixed"] = boot(per["D3g"] - per["D3f"])
res["by_day"] = {a: [boot(np.nanmean(arms[a][0][:, d] - arms["hardcbB"][0][:, d], axis=1))[0] for d in range(3)] for a in ("D3f", "D3g")}
res["per_person"] = {a: np.round(per[a] - per["hardcbB"], 2).tolist() for a in ("D3f", "D3g")}
for a in ("D3f", "D3g"):
    M = arms[a][1]
    rt, ac, sh, un = (np.nansum(M[..., i]) for i in (4, 5, 6, 7))
    res.setdefault("drift", {})[a] = dict(routed=float(rt), active_share=float(ac / max(rt, 1)), short=float(sh), uncorroborated=float(un),
                                          mean_r=float(np.nansum(M[..., 9]) / max(ac, 1)), max_logit_move=float(np.nanmax(M[..., 10])))
    nwin = np.nansum(M[..., 4]) or 1
sec = {a: arms[a][1][..., 3] for a in arms}
res["seconds_per_stream"] = {a: float(np.nanmean(s)) for a, s in sec.items()}
res["recognised"] = float(np.nanmean(arms["hardcbB"][1][..., 0] >= 0))
# R4 matched (same two orders), hardcbB from B10 for the same orders as a consistency check
r4, hb = b10_arm("hardcbB_R4"), b10_arm("hardcbB")
res["R4_matched_vs_B10_hardcbB"] = boot(np.nanmean(r4 - hb, axis=(1, 2)))
res["B10_hardcbB_vs_D3run_hardcbB_abs_diff"] = float(np.nanmean(hb) - np.nanmean(arms["hardcbB"][0]))
Path(dest).write_text(json.dumps(res, indent=1))
print(json.dumps({k: v for k, v in res.items() if k != "per_person"}, indent=1))
