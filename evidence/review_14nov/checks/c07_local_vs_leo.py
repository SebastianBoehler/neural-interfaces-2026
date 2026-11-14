"""Every regular v6f file (gC5, labels L) and v6.1 file that exists both locally (runs/) and as a Leonardo copy (runs/leo_npz):
is the local one still the regular result? V6Fr - P0r, six-base mean, and the local file's time. Ledger s188 form B per order:
current eye net -> form B (pseudo-labels) -> true-label ceiling, for comparison."""
import glob, os, time
import numpy as np
REPO = os.path.abspath(__file__ + "/../../../..")
FORMB = {"within0": (1.39, 1.51, 1.70), "within1": (1.49, 1.68, 1.84), "within2": (0.32, 0.42, 0.58), "within3": (1.02, 1.18, 1.31),
         "full0": (1.27, 1.24, 1.35), "full1": (0.62, 0.69, 0.82), "runs0": (1.35, 1.47, 1.44), "runs1": (0.91, 0.94, 1.06)}


def gain(f):
    z = np.load(f); b = sorted({k.split("|")[0] for k in z.files}); return np.mean([z[f"{bn}|V6Fr"] - z[f"{bn}|P0r"] for bn in b], 0)


for f in sorted(glob.glob(f"{REPO}/runs/leo_npz/shufcombo_*.npz")):
    n = os.path.basename(f); a = f"{REPO}/runs/{n}"
    if not os.path.exists(a):
        continue
    ga, gb = gain(a), gain(f); o = n.split("_")[1]; bad = np.abs(ga - gb).max() > 0.11
    print(f"{'DIFFERS' if bad else 'same   '} {n[10:]:58s} local {ga.mean():+.2f} (written {time.strftime('%d %H:%M', time.gmtime(os.path.getmtime(a)))} UTC) Leonardo {gb.mean():+.2f} "
          f"max person diff {np.abs(ga - gb).max():.2f}" + (f" | s188 form B row for {o}: {FORMB[o]}" if bad and o in FORMB else ""))
print("\nlocal-only regular files (no Leonardo copy to compare with):")
for a in sorted(glob.glob(f"{REPO}/runs/shufcombo_*_ocu_gC5_all_ge_hb_ts_gp0_ttt18*L.npz")):
    if not os.path.exists(f"{REPO}/runs/leo_npz/{os.path.basename(a)}"):
        print(f"   {os.path.basename(a)[10:]:58s} {gain(a).mean():+.2f} written {time.strftime('%d %H:%M', time.gmtime(os.path.getmtime(a)))} UTC")
