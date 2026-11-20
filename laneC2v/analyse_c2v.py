"""Lane C2v analysis: arms vs own base, both orders; people 80 % (orders fixed) and people+orders 80 % intervals (10,000 draws); numpy only. usage: analyse_c2v.py"""
import json
import numpy as np

rng = np.random.default_rng(461)
pp = rng.integers(0, 10, (10000, 10)); oo = rng.integers(0, 2, (10000, 2))
for base in ("hardcb", "ds_l5"):
    d = json.loads(open(f"cells_{base}.json").read())
    C = d["cells"]; people = sorted(C["full0"]["base"]); n = np.array([C["full0"]["base"][p][1] for p in people], float)
    S = lambda o, a: np.array([C[o][a][p][0] for p in people])
    arms = [a for a in C["full0"] if a not in ("base", "_selected")]
    def con(a, b, ords=("full0", "within0")):
        x = np.stack([S(o, a) - S(o, b) for o in ords])
        pt = 100 * x.mean(0).sum() / n.sum()
        ppl = np.percentile([100 * x.mean(0)[pp[i]].sum() / n[pp[i]].sum() for i in range(10000)], (10, 90))
        po = np.percentile([100 * x[oo[i] % len(ords)][:, pp[i]].mean(0).sum() / n[pp[i]].sum() for i in range(10000)], (10, 90))
        return pt, ppl, po
    print(f"== {base}: ms/window joint(other) full0 {d['ms_per_window']['full0|joint|other']:.4f}")
    print("arm | both [people80] [people+orders80] | full0 | within0 | P04 P10 (pp, both orders)")
    for a in arms:
        b, pl, po = con(a, "base")
        f = con(a, "base", ("full0",))[0]; w = con(a, "base", ("within0",))[0]
        x = np.mean([(S(o, a) - S(o, "base")) / n * 100 for o in ("full0", "within0")], axis=0)
        print(f"{a:14s} {b:+.2f} [{pl[0]:+.2f},{pl[1]:+.2f}] [{po[0]:+.2f},{po[1]:+.2f}] | {f:+.2f} | {w:+.2f} | {x[people.index('P04')]:+.2f} {x[people.index('P10')]:+.2f}")
    for s in ("other", "same", "both"):
        pm = np.mean([[S(o, f"perm{k}|{s}") for k in range(3)] for o in ("full0", "within0")], axis=(0, 1))  # people sums averaged over perms and orders
        j = np.mean([S(o, f"joint|{s}") for o in ("full0", "within0")], axis=0); bs = np.mean([S(o, "base") for o in ("full0", "within0")], axis=0)
        bl = np.mean([S(o, f"blind|{s}") for o in ("full0", "within0")], axis=0)
        print(f"{s}: joint-base {100*(j-bs).sum()/n.sum():+.2f}  joint-permmean {100*(j-pm).sum()/n.sum():+.2f}  joint-blind {100*(j-bl).sum()/n.sum():+.2f}  perm-mean-base {100*(pm-bs).sum()/n.sum():+.2f}")
    print("selected", {o: C[o]["_selected"]["chosen"] for o in C})
