"""Which files morning_table.py pairs per order (replicates its PAIRS loop, read-only) and what a draw-0-only pairing would give."""
import glob, os
import numpy as np
REPO = os.path.abspath(__file__ + "/../../../.."); NPZ = REPO + "/runs/leo_npz"


def one(pat):
    fs = glob.glob(pat); assert len(fs) <= 1, fs; return fs[0] if fs else None


def mb(f, arm):
    z = np.load(f); b = sorted({k.split("|")[0] for k in z.files}); assert len(b) == 6; return np.mean([z[f"{bn}|{arm}"] for bn in b], 0)


PAIRS, HIST = {}, {}
for d, tag in ((REPO + "/runs", "QF"), (NPZ, "RF"), (NPZ, "KF")):
    for f in sorted(glob.glob(f"{d}/shufcombo_*_ocu_gC5_all_ge_hb_ts_gp0_ttt18*L.npz")):
        o = os.path.basename(f).split("_")[1]; g = one(f"{d}/shufcombo_{o}_ocu_s15_all_ge_hb_ts_gp0_ttt18{o}{tag}.npz")
        HIST.setdefault((d.replace(REPO, ""), tag, o), []).append(os.path.basename(f))
        if g is not None:
            PAIRS[o] = (f, g)
print("globs matching more than one v6f file for one order (last one wins if a pair exists):")
for k, v in HIST.items():
    if len(v) > 1:
        print("  ", k, v)
print("\nfinal pairs:")
for o, (f, g) in sorted(PAIRS.items()):
    print(f"  {o:9s} v6f {f.replace(REPO, '')[1:]}\n            v6.1 {g.replace(REPO, '')[1:]}")
print("\nlabel form of the v6.1 column per order:", {o: g[-6:-4] for o, (f, g) in sorted(PAIRS.items())})
