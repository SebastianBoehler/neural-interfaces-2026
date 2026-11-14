"""Read-only: nested choice among the final new-people arms (none, fallback, v6f, v6.1): pick on five people (mean over
the scheme's orders), score the pick on the other five, all 252 ordered splits. Covers the last choice only."""
import glob, itertools, os
import numpy as np
REPO = os.path.abspath(__file__ + "/../../../.."); NPZ = REPO + "/runs/leo_npz"


def mb(f, arm):
    z = np.load(f); b = sorted({k.split("|")[0] for k in z.files}); return np.mean([z[f"{bn}|{arm}"] for bn in b], 0)


PAIRS = {}
for d, tag in ((REPO + "/runs", "QF"), (NPZ, "RF"), (NPZ, "KF")):
    for f in sorted(glob.glob(f"{d}/shufcombo_*_ocu_gC5_all_ge_hb_ts_gp0_ttt18*L.npz")):
        o = os.path.basename(f).split("_")[1]; g = glob.glob(f"{d}/shufcombo_{o}_ocu_s15_all_ge_hb_ts_gp0_ttt18{o}{tag}.npz")
        if g:
            PAIRS[o] = (f, g[0])
for sch in ("within", "full", "runs"):
    os_ = [o for o in PAIRS if o.startswith(sch)]
    A = {"none": np.zeros(10), "fallback": np.mean([mb(PAIRS[o][0], "V6r") - mb(PAIRS[o][0], "P0r") for o in os_], 0),
         "v6f": np.mean([mb(PAIRS[o][0], "V6Fr") - mb(PAIRS[o][0], "P0r") for o in os_], 0),
         "v6.1": np.mean([mb(PAIRS[o][1], "V6Fr") - mb(PAIRS[o][1], "P0r") for o in os_], 0)}
    held, picks = [], {}
    for tr in itertools.combinations(range(10), 5):
        te = [i for i in range(10) if i not in tr]; k = max(A, key=lambda a: A[a][list(tr)].mean()); picks[k] = picks.get(k, 0) + 1
        held.append(A[k][te].mean())
    print(f"{sch}: in-sample best {max(A, key=lambda a: A[a].mean())} {max(v.mean() for v in A.values()):+.2f} | nested held-out {np.mean(held):+.2f} "
          f"[10th {np.percentile(held, 10):+.2f}, 90th {np.percentile(held, 90):+.2f}] | picks {picks}")
