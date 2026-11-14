"""s262 scorer: routed minus tail per person (ER, ERT; also ERoff, P, PT), interval over orders, median over the eight
non-P04/P10 people, count >= +2, sealed-weighted dev level. Usage: python score262.py runs/pview/routed262_<orders>.npz > evidence/returning10_14nov/s262.txt"""
import sys
import numpy as np
z = np.load(sys.argv[1]); PEOPLE = [f"P{i:02d}" for i in range(1, 11)]; OTH = [i for i, q in enumerate(PEOPLE) if q not in ("P04", "P10")]
ORDERS = sorted({k.split("|")[0] for k in z.files}); BASES = sorted({k.split("|")[1] for k in z.files}); assert len(BASES) == 6, BASES
ARMS = [a for a in ("tail0", "ERoff", "P", "PT", "ER", "ERT") if f"{ORDERS[0]}|{BASES[0]}|{a}" in z.files]
M = {a: np.array([[z[f"{o}|{b}|{a}"] for b in BASES] for o in ORDERS]).mean(1) for a in ARMS}  # (orders, 10): six-base mean
print(f"s262: routed personal path on the within-session proxy, oracle recognition, ro = 1 on online rows; {len(ORDERS)} orders ({', '.join(ORDERS)}) x 6 bases")
print(f"tail0 (= PV6 base, asserted per order and base): mean {M['tail0'].mean():.2f}; " + " ".join(f"{q} {v:.1f}" for q, v in zip(PEOPLE, M['tail0'].mean(0))))
rng = np.random.default_rng(0)
for a in [a for a in ARMS if a != "tail0"]:
    d = M[a] - M["tail0"]; m = d.mean(0)
    print(f"\n{a} minus tail0: mean over ten {m.mean():+.2f}; median over the eight (non-P04/P10) {np.median(m[OTH]):+.2f}; count >= +2 among the eight {(m[OTH] >= 2).sum()}/8; "
          f"< +2: {(m[OTH] < 2).sum()}/8; >= +2 in all ten {(m >= 2).sum()}/10")
    for i, q in enumerate(PEOPLE):
        lo, hi = np.percentile(d[:, i], [10, 90]); print(f"   {q} {m[i]:+6.2f} [{lo:+.2f}, {hi:+.2f}] over orders (orders > 0: {(d[:, i] > 0).sum()}/{len(ORDERS)})")
    oo = np.array([np.median(d[rng.integers(0, len(ORDERS), len(ORDERS))][:, OTH].mean(0)) for _ in range(4000)])
    print(f"   median over the eight, 80 % interval over orders: [{np.percentile(oo, 10):+.2f}, {np.percentile(oo, 90):+.2f}]")
    pp = np.array([np.median(d[:, rng.choice(OTH, 8)].mean(0)) for _ in range(4000)])
    print(f"   median over the eight, 80 % interval resampling people (and orders fixed): [{np.percentile(pp, 10):+.2f}, {np.percentile(pp, 90):+.2f}]")
if "ER" in ARMS:
    print("\nreference: ER minus ERoff (the personal path alone, ER inputs held): " + " ".join(f"{q} {v:+.2f}" for q, v in zip(PEOPLE, (M['ER'] - M['ERoff']).mean(0))) if "ERoff" in ARMS else "")
# sealed-weighted dev level, §249's calibration: J = 0.566 new + 0.434 returning; new 77.8; returning = 0.2 P04 + 0.8 P(other returning)
NEW, P04, P10 = 77.8, 70.0, 89.6; NEW_LVL = np.array([76.7, 89.1, 73.1, 52.6, 75.6, 83.4, 88.8, 86.2, 82.6, 82.5])  # §249 r15 new-people dev levels
print("\nsealed-weighted dev level (§249: J = 0.566 x new 77.8 + 0.434 x returning; returning = 0.2 P04 70.0 + 0.8 x other returning):")
J = lambda ret: 0.566 * NEW + 0.434 * (0.2 * P04 + 0.8 * ret)
print(f"   (a) other returning = P10 89.6 (assumption): returning {0.2 * P04 + 0.8 * P10:.2f}, J {J(P10):.2f}")
for a in [x for x in ("ER", "ERT") if x in ARMS]:
    lv = M[a].mean(0)[OTH].mean()  # proxy absolute level of the eight, routed path (within-session: optimistic, no between-session shift)
    gn = (M[a] - M["tail0"]).mean(0)[OTH]; lv2 = (NEW_LVL[OTH] + gn).mean()
    print(f"   (b) {a}, measured ten-person spread, proxy absolute level of the eight {lv:.2f}: returning {0.2 * P04 + 0.8 * lv:.2f}, J {J(lv):.2f}")
    print(f"   (c) {a}, §249 new-people level of the eight ({NEW_LVL[OTH].mean():.2f}) + proxy gain {gn.mean():+.2f} = {lv2:.2f}: returning {0.2 * P04 + 0.8 * lv2:.2f}, J {J(lv2):.2f}")
