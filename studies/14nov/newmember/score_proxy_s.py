"""s281 scoring on the ten-person routed proxy (proxy_s.py outputs runs/pview/routed262_S<k>_<orders>.npz, keys order|base|arm, arms tail0 and ERT, (10,) per person).
S0 (ZERO=1: new slot at weight 0) must equal s262's own tail0 / ERT (runs/pview/routed262_<orders>.npz) to 1e-9. Per arm S1-S5: ERT (and tail0) minus S0's, six-base mean, mean over
the 8 orders (schemes within / full / runs equal weight), per person; 80 % interval over people; ten-person mean next to P04, P10 and the returning weighting 0.2 P04 + 0.8 P10.
Usage: python studies/14nov/newmember/score_proxy_s.py > evidence/newmember_15nov/s281_proxy.txt"""
import os
import numpy as np
REPO = os.path.abspath(__file__ + "/../../../.."); O = "within0-within1-within2-within3-full0-full1-runs0-runs1"; P = f"{REPO}/runs/pview/routed262_"
PEOPLE = [f"P{i:02d}" for i in range(1, 11)]; i04, i10 = PEOPLE.index("P04"), PEOPLE.index("P10")
Z = {c: np.load(f"{P}{c}_{O}.npz") for c in ("S0", "S1", "S2", "S3", "S4", "S5")}; ref = np.load(f"{P}{O}.npz")
ORD = sorted({k.split("|")[0] for k in Z["S0"].files}); BAS = sorted({k.split("|")[1] for k in Z["S0"].files}); assert len(ORD) == 8 and len(BAS) == 6, (ORD, BAS)
for a in ("tail0", "ERT"):  # zero-change: S0 == s262 reference
    assert all(np.abs(Z["S0"][f"{o}|{b}|{a}"] - ref[f"{o}|{b}|{a}"]).max() < 1e-9 for o in ORD for b in BAS), f"S0 {a} differs from s262"
for c in Z:
    assert set(Z[c].files) == set(Z["S0"].files), c
SCH = ("within", "full", "runs"); rng = np.random.default_rng(0)
M = lambda c, a, bs=BAS: np.array([[Z[c][f"{o}|{b}|{a}"] for b in bs] for o in ORD]).mean(1)  # (orders, 10)


def sch(d):  # mean over orders within scheme, schemes equal weight -> (10,)
    return np.mean([d[[i for i, o in enumerate(ORD) if o.rstrip("0123456789") == s]].mean(0) for s in SCH], 0)


print(f"s276 ten-person routed proxy, {len(ORD)} orders x 6 bases; S0 == s262 tail0/ERT (asserted 1e-9). S0 level: tail0 {sch(M('S0', 'tail0')).mean():.2f}, ERT {sch(M('S0', 'ERT')).mean():.2f}")
for c in ("S1", "S2", "S3", "S4", "S5"):
    for a in ("ERT", "tail0"):
        d = M(c, a) - M("S0", a); m = sch(d)
        po = np.array([m[rng.integers(0, 10, 10)].mean() for _ in range(4000)])
        print(f"\n{c} {a} minus S0: ten-person mean {m.mean():+.2f} [people 80 %: {np.percentile(po, 10):+.2f}, {np.percentile(po, 90):+.2f}] | P04 {m[i04]:+.2f} P10 {m[i10]:+.2f} 0.2/0.8 {0.2 * m[i04] + 0.8 * m[i10]:+.2f} | people up {(m > 0).sum()}/10, worse than -0.2: {(m < -0.2).sum()}/10")
        print("   per person: " + " ".join(f"{q} {v:+.2f}" for q, v in zip(PEOPLE, m)) + " | schemes " + " ".join(f"{s} {d[[i for i, o in enumerate(ORD) if o.rstrip('0123456789') == s]].mean():+.2f}" for s in SCH))
