"""s266 part (b) scorer: nested-by-person pick over the grid of proxy266.py (MODE=grid), real gate; nested tuned minus shipped.
Rule (amendment 1): per held-out person the argmax of the other nine's mean BA, the shipped config unless it wins by >= MARGIN.
Usage: python score266b.py 'runs/pview/s266_grid_*.npz' (glob: the per-order-group files) [runs/pview/gate266.npz]"""
import sys, re, glob
import numpy as np
z = {}
for f_ in sorted(glob.glob(sys.argv[1])):
    z.update(dict(np.load(f_)))
ZF = list(z)
PEOPLE = [f"P{i:02d}" for i in range(1, 11)]; OTH = [i for i, q in enumerate(PEOPLE) if q not in ("P04", "P10")]; MARGIN = 0.05
ORDERS = sorted({k.split("|")[0] for k in ZF}); BASES = sorted({k.split("|")[1] for k in ZF}); CFG = sorted({k.split("|")[2] for k in ZF})
SHIP = "g3_k20_t0.26_r0.05_b0.1"; assert SHIP in CFG and len(CFG) == 432, (len(CFG), SHIP in CFG)
Cord = np.array([[np.mean([z[f"{o}|{b}|{c}"] for b in BASES], 0) for o in ORDERS] for c in CFG])  # (config, order, person)
M = Cord.mean(1); si = CFG.index(SHIP); par = lambda c: dict(zip("gktrb", map(float, re.findall(r"[-\d.]+", c))))  # noqa: E731
print(f"s266 (b): grid of {len(CFG)} configs x {len(ORDERS)} orders x {len(BASES)} bases (real bio gate); shipped {SHIP}; shipped mean over ten {M[si].mean():.2f}")
print("\none-at-a-time around shipped (mean over ten minus shipped; per person range):")
for name, key, vals in (("own-space beta", "b", (0.05, 0.10, 0.20, 0.30)), ("head gain G3", "g", (2, 3, 4, 6)), ("TSEM kappa", "k", (10, 20, 40)), ("tau", "t", (0.22, 0.26, 0.30)), ("ramp", "r", (0.03, 0.05, 0.08))):
    row = []
    for v in vals:
        p = dict(par(SHIP)); p[key] = v; c = f"g{p['g']:g}_k{p['k']:g}_t{p['t']:g}_r{p['r']:g}_b{p['b']:g}"; d = M[CFG.index(c)] - M[si]
        row.append(f"{v:g}: {d.mean():+.2f} (eight median {np.median(d[OTH]):+.2f}, min {d.min():+.2f})")
    print(f"  {name:15s} " + " | ".join(row))


def nested(allowed):
    ids = [i for i, c in enumerate(CFG) if c in allowed]; picks = []
    for q in range(10):
        nine = [i for i in range(10) if i != q]; sc = M[ids][:, nine].mean(1); best = ids[int(np.argmax(sc))]
        picks.append(best if M[best, nine].mean() - M[si, nine].mean() >= MARGIN else si)
    return picks


def report(label, allowed):
    picks = nested(allowed); d = np.array([M[picks[q], q] - M[si, q] for q in range(10)]); rng = np.random.default_rng(0)
    dord = np.array([[Cord[picks[q], o, q] - Cord[si, o, q] for q in range(10)] for o in range(len(ORDERS))])
    oo = np.array([dord[rng.integers(0, len(ORDERS), len(ORDERS))].mean() for _ in range(4000)]); pp = np.array([d[rng.integers(0, 10, 10)].mean() for _ in range(4000)])
    print(f"\n{label}: nested tuned minus shipped, mean over ten {d.mean():+.3f}; 80 % over orders [{np.percentile(oo, 10):+.2f}, {np.percentile(oo, 90):+.2f}], 80 % resampling people [{np.percentile(pp, 10):+.2f}, {np.percentile(pp, 90):+.2f}]; "
          f"people up {(d > 0).sum()}/10 (down {(d < 0).sum()}); median of the eight {np.median(d[OTH]):+.2f}; P04 {d[0]:+.2f} P10 {d[9]:+.2f}")
    print("  per person (pick -> delta): " + "; ".join(f"{PEOPLE[q]} {CFG[picks[q]] if picks[q] != si else 'shipped'} {d[q]:+.2f}" for q in range(10)))
    print(f"  picks differing from shipped: {sum(p != si for p in picks)}/10; distinct picks {len(set(picks))}")
    return picks, d


picks, d = report("all constants", set(CFG))
report("tau / ramp held at shipped", {c for c in CFG if par(c)["t"] == 0.26 and par(c)["r"] == 0.05})
report("own-space beta and head gain only", {c for c in CFG if par(c)["t"] == 0.26 and par(c)["r"] == 0.05 and par(c)["k"] == 20})
amb = M.mean(1); b = int(np.argmax(amb)); print(f"\nin-sample (not nested, optimistic) best config on all ten: {CFG[b]} {amb[b] - amb[si]:+.3f} over shipped")
np.savez(sys.argv[1].replace("*", "").replace(".npz", "_picks.npz"), picks=np.array([CFG[p] for p in picks]), delta=d)
if len(sys.argv) > 2:  # false-route windows of the gate at the picked tau / ramp against shipped
    g = np.load(sys.argv[2]); fk = [k for k in g.files if k.startswith("false|") and k.endswith("|s")]
    for t in sorted({par(CFG[p])["t"] for p in picks} | {0.26}):
        n = sum(int(((g[k[:-1] + "j"] == len(g[k[:-1] + "names"]) - 1) & (g[k] > t)).sum()) for k in fk)
        print(f"false-route windows (all session pairs, 8 orders) at tau {t:g}: {n}")
