"""shufcombo adds (delivery - own recording rebuild) to stored members without clipping (core1, Hilbert via slot 0, teacher slot 10).
How close is each rebuild to the stored member, and do the patched members leave [0, 1]?"""
import glob, os
import numpy as np
REPO = os.path.abspath(__file__ + "/../../../.."); D = REPO + "/runs/replay_data"
Z = np.load(D + "/inputs.npz", allow_pickle=True); B = np.load(D + "/bases.npz", allow_pickle=True); p = "all_loso"
c1 = B[f"B2_pool|core1|{p}"]; ps = Z[f"B2_pool|Ps|{p}"]; hb = 2 * ps[0] - c1
for name, files, ref in (("core1", sorted(glob.glob(REPO + "/runs/coremember*.npz")), c1), ("hilbert", sorted(glob.glob(REPO + "/runs/hilbertmember*.npz")), hb),
                         ("teacher", sorted(glob.glob(REPO + "/runs/teachshuf*.npz")), ps[10])):
    for f in files[:2]:
        z = np.load(f); rec = z[f"rec|{p}"]
        print(f"{name} {os.path.basename(f)}: rebuild vs stored max |diff| {np.abs(rec - ref).max():.3f}, mean {np.abs(rec - ref).mean():.4f}, argmax agreement {np.mean(rec.argmax(1) == ref.argmax(1)):.4f}")
        for k in [k for k in z.files if k.endswith("|" + p) and not k.startswith("rec")][:2]:
            patched = ref + (z[k] - rec)
            print(f"    {k}: patched member min {patched.min():+.3f} max {patched.max():+.3f}; rows with a negative entry {np.mean(patched.min(1) < 0):.4f}; "
                  f"argmax(patched) == argmax(delivery stream) {np.mean(patched.argmax(1) == z[k].argmax(1)):.4f}")
