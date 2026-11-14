"""DOSE x OPERATING-CURVE MAP (ledger s252 amendment 2). (a) c x aE gain over the r15 chain per (noise kind, n, purity):
person mean over sessions of the mean over pairs of the mean over the six bE orders x six bases (schemes equal), then the
ten-person mean. (b) every operating-curve point (scheme, t, ranking, k, purity q) looked up in (a)'s chain-like grid by
bilinear interpolation at (n = k, p = q), k clipped to [20, n_all], q clipped to [75, 100] (below 75 % the 75 % value is an
upper bound). Verdict: any point with gain >= +1?
Usage: python studies/14nov/personal_ceiling/score_dose.py > evidence/personal_ceiling_14nov/dose_map.txt"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np  # noqa: E402
import splits as S  # noqa: E402
from dose import PUR  # noqa: E402

R = S.REPO; m = S.meta(); NS = ("20", "40", "70", "100", "all")
SIX = ("within0", "within1", "full0", "full1", "runs0", "runs1"); WO = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5])  # two orders per scheme
PAIRS = [(s, n, t, tr) for s in S.sessions(m) for n, tr, t in S.pairs(m, s)]


def chain():
    out = []
    for o in SIX:
        z = np.load(R / f"runs/chainout/s242r15_{o}.npz"); out.append([z[k].astype(np.float64) for k in sorted(z.files) if k.endswith("|V6Fr")])
    return out


def person(vals):
    v = {}
    for (s, _, _, _), x in zip(PAIRS, vals):
        v.setdefault(s, []).append(x)
    return np.array([np.mean([np.mean(x) for s, x in v.items() if s.startswith(p)]) for p in S.PEOPLE])


def fused(CH, Q):
    """Q: list of personal outputs per pair (None = chain alone) -> person scores"""
    vals = []
    for (s, n, t, _), q in zip(PAIRS, Q):
        vals.append(np.average([np.mean([S.cell_score(m, t, S.fuse(c[t], q) if q is not None else c[t]) for c in co]) for co in CH], weights=WO))
    return person(vals)


def main():
    CH = chain(); base = fused(CH, [None] * len(PAIRS)); nall = np.mean([len(tr) for *_, tr in PAIRS]); G = {}
    print(f"(a) DOSE-RESPONSE, true labels: c x aE gain over the r15 chain (six orders; chain {base.mean():.2f}); n 'all' = {nall:.0f} windows")
    for kind in ("chain", "rand"):
        print(f"  noise {kind}: rows n, columns purity " + " ".join(f"{p:>6d}" for p in PUR) + "   | aE alone at 100 / 75")
        for ns in NS:
            z = np.load(S.OUT / f"dose_{kind}_{ns}.npz"); row = []
            for p in PUR:
                g = fused(CH, [z[f"{s}|{n}|p{p}"] for s, n, _, _ in PAIRS]) - base; G[kind, ns, p] = g; row.append(g.mean())
            al = [person([S.cell_score(m, t, z[f"{s}|{n}|p{p}"]) for s, n, t, _ in PAIRS]).mean() for p in (100, 75)]
            print(f"    n {ns:>4s}  " + " ".join(f"{x:+6.2f}" for x in row) + f"   | {al[0]:.1f} / {al[1]:.1f}")
        for ns in NS:
            ok = [p for p in PUR if G[kind, ns, p].mean() >= 1]; print(f"    +1 contour, n {ns}: purity >= {min(ok) if ok else 'none'} %")
    nx = np.array([20, 40, 70, 100, nall]); px = np.array(PUR[::-1], float); grid = np.array([[G["chain", ns, int(p)].mean() for p in px] for ns in NS])
    look = lambda k, q: float(np.interp(np.clip(q, 75, 100), px, [np.interp(np.clip(k, 20, nall), nx, grid[:, j]) for j in range(len(px))]))  # noqa: E731
    oc = np.load(S.OUT / "opcurve.npz"); best = []
    print("\n(b) OPERATING CURVE in (a)'s chain-like grid: best point per scheme, t, ranking (k, purity, interpolated gain)")
    for key in oc.files:
        sc, t, r = key.split("|"); pts = [(int(k), q, look(k, q)) for k, q, _ in oc[key]]; b = max(pts, key=lambda x: x[2]); best.append((b[2], key, b))
        print(f"  {sc:6s} t={'end' if int(t) > 1000 else t:>3s} {r:5s}: k {b[0]:3d} purity {b[1]:.1f} -> {b[2]:+.2f}")
    top = max(best); print(f"\nVERDICT: best point {top[1]} k {top[2][0]} purity {top[2][1]:.1f} gain {top[0]:+.2f} -> " + ("falls in the >= +1 region" if top[0] >= 1 else "no point reaches +1: the class is sized and closed for this feature set"))


if __name__ == "__main__":
    main()
