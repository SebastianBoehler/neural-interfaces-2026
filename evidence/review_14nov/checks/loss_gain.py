"""Read-only: is the candidate's new-people gain tied to the baseline's dev order loss? Per person, per scheme."""
import glob, os
import numpy as np
REPO = os.path.abspath(__file__ + "/../../../.."); NPZ = REPO + "/runs/leo_npz"


def mb(f, arm):
    z = np.load(f); b = sorted({k.split("|")[0] for k in z.files})
    return np.mean([z[f"{bn}|{arm}"] for bn in b], 0) if f"{b[0]}|{arm}" in z.files else None


F = {}
for d in (REPO + "/runs", NPZ):
    for f in sorted(glob.glob(f"{d}/shufcombo_*_ocu_gC5_all_ge_hb_ts_gp0_ttt18*L.npz")):
        F[os.path.basename(f).split("_")[1]] = f
print("arms in file:", sorted({k.split("|")[1] for k in np.load(next(iter(F.values()))).files}))
for sch in ("within", "full", "runs"):
    os_ = [o for o in F if o.startswith(sch)]
    loss = np.mean([mb(F[o], "P0r") - mb(F[o], "P0rec") for o in os_], 0)
    gain = np.mean([mb(F[o], "V6Fr") - mb(F[o], "P0r") for o in os_], 0)
    net = np.mean([mb(F[o], "V6Fr") - mb(F[o], "P0rec") for o in os_], 0)   # candidate shuffled vs v5.1b in recording order
    rec = [mb(F[o], "V6Frec") - mb(F[o], "P0rec") for o in os_ if mb(F[o], "V6Frec") is not None]
    r = np.corrcoef(-loss, gain)[0, 1]; b, a = np.polyfit(-loss, gain, 1)
    print(f"{sch}: {len(os_)} orders | v5.1b order loss {loss.mean():+.2f} | v6f gain {gain.mean():+.2f} | corr(-loss, gain) over 10 people {r:+.2f} | "
          f"gain = {a:+.2f} + {b:.2f} x loss recovered | v6f shuffled vs v5.1b recording {net.mean():+.2f}"
          + (f" | v6f gain in recording order {np.mean(rec):+.2f} ({len(rec)} files; per person " + " ".join(f"{x:+.1f}" for x in np.mean(rec, 0)) + ")" if rec else " | no recording-order arm"))
    # per-order relation across orders: do orders with a larger baseline loss show a larger gain?
    lo = [float((mb(F[o], "P0r") - mb(F[o], "P0rec")).mean()) for o in os_]; ga = [float((mb(F[o], "V6Fr") - mb(F[o], "P0r")).mean()) for o in os_]
    print(f"   across orders: corr(-loss, gain) {np.corrcoef(-np.array(lo), ga)[0, 1]:+.2f}; loss range {min(lo):+.2f}..{max(lo):+.2f}, gain range {min(ga):+.2f}..{max(ga):+.2f}")
