"""s207 leverage and stage 2 read the unedited base from local runs/. For full0 and runs0 that file holds the s188 form-B
true-label run (c07). Recomputes both tables against the Leonardo copy of the regular file (identical for within0)."""
import glob, os
import numpy as np
REPO = os.path.abspath(__file__ + "/../../../.."); ORD = ("within0", "full0", "runs0")


def m(f, arm):
    z = np.load(f); b = sorted({k.split("|")[0] for k in z.files}); assert len(b) == 6; return np.mean([z[f"{x}|{arm}"] for x in b], 0)


def bases(o):
    n = f"shufcombo_{o}_ocu_gC5_all_ge_hb_ts_gp0_ttt18{o}L.npz"; return f"{REPO}/runs/{n}", f"{REPO}/runs/leo_npz/{n}"


print("base check (P0r must be equal; V6Fr local - Leonardo):")
for o in ORD:
    a, b = bases(o); print(f"  {o}: P0r max diff {np.abs(m(a, 'P0r') - m(b, 'P0r')).max():.3f}; V6Fr mean local - Leonardo {np.mean(m(a, 'V6Fr') - m(b, 'V6Fr')):+.3f}")
print("\nLEVERAGE per point (+5), as written (local base) -> against the regular file; rule: >= 0.05 on two orders stage 2, < 0.03 everywhere no lever")
for c in [f"s{k}" for k in range(11)] + ["c1", "rest", "eye"]:
    old, new = [], []
    for o in ORD:
        a, b = bases(o); f = a[:-4] + f"_syn{c}5.npz"
        old.append(np.mean(m(f, "V6Fr") - m(a, "V6Fr")) / 5); new.append(np.mean(m(f, "V6Fr") - m(b, "V6Fr")) / 5)
    rule = lambda g: "stage 2" if sum(x >= 0.05 for x in g) >= 2 else ("no lever" if max(g) < 0.03 else "undecided")
    print(f"  {c:5s} written " + " ".join(f"{x:+.3f}" for x in old) + f" mean {np.mean(old):+.3f} [{rule(old)}] -> " + " ".join(f"{x:+.3f}" for x in new) + f" mean {np.mean(new):+.3f} [{rule(new)}]")
print("\nSTAGE 2 chain change (V6Fr), as written -> against the regular file")
for f in sorted(glob.glob(f"{REPO}/runs/wsp/final_*.npz")):
    tag = os.path.basename(f)[6:-4]; old, new = [], []
    for o in ORD:
        a, b = bases(o); g = a[:-4] + f"_ox{tag}.npz"
        if os.path.exists(g):
            old.append(np.mean(m(g, "V6Fr") - m(a, "V6Fr"))); new.append(np.mean(m(g, "V6Fr") - m(b, "V6Fr")))
    if len(old) == 3:
        print(f"  {tag:12s} written " + " ".join(f"{x:+.2f}" for x in old) + f" mean {np.mean(old):+.2f} -> " + " ".join(f"{x:+.2f}" for x in new) + f" mean {np.mean(new):+.2f}")
