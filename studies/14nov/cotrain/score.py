"""Scorer for ledger s258 (eye-net co-training labels). c x aE gain over the r15 chain per arm, 20 orders, §252 splits; splits
with a fallback flag (class < 2 kept windows) use the chain alone. Reuses personal_ceiling/score_pl.py (person_scores, wmean, boot).
Also: kept-label purity and n per arm (non-fallback splits, mean over splits), fallback share, and the §252 dose-map value
(chain-like grid of evidence/personal_ceiling_14nov/dose_map.txt, bilinear in (n, purity), clipped to the grid) at that (n, purity).
Usage: python studies/14nov/cotrain/score.py > evidence/cotrain_14nov/score.txt"""
import re, sys
from pathlib import Path
REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "studies/14nov/personal_ceiling")); sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np  # noqa: E402
import score_pl as P  # noqa: E402
import splits as S  # noqa: E402
from eyelab import ARMS, OUT  # noqa: E402


def dose_grid():
    L = open(REPO / "evidence/personal_ceiling_14nov/dose_map.txt").read().split("noise rand")[0]
    rows = [[float(x) for x in re.findall(r"[+-]\d+\.\d+", ln.split("|")[0])] for ln in L.splitlines() if re.match(r"\s+n\s+(\d+|all)\s", ln)]
    return np.array(rows)  # (5, 6): n 20 40 70 100 all, purity 100..75


def look(G, k, q, nall=129):
    px = np.array([75, 80, 85, 90, 95, 100.]); nx = np.array([20, 40, 70, 100, nall])
    col = [np.interp(np.clip(k, 20, nall), nx, G[:, 5 - j]) for j in range(6)]; return float(np.interp(np.clip(q, 75, 100), px, col))


def main():
    R = P.R; ALL = P.ALL; CH = {o: [np.load(R / f"runs/chainout/s242r15_{o}.npz")[k].astype(np.float64) for k in sorted(np.load(R / f"runs/chainout/s242r15_{o}.npz").files) if k.endswith("|V6Fr")] for o in ALL}
    Z = {o: np.load(OUT / f"ec_{o}.npz") for o in ALL}; G = dose_grid(); base = P.person_scores(CH, ALL, lambda o, s, n: []); cb = P.wmean(base, ALL)
    print(f"r15 chain on the test runs, ten-person mean {cb.mean():.2f}; 20 orders; fallback = chain alone\n")
    print("arm   n  purity  fallback | dose-map | c x aE gain [80 % people x orders] people up | within / full / runs | per person P01..P10")
    for arm in ARMS:
        fb = np.array([Z[o][f"W|{s}|{n}|{arm}|fb"][0] for o in ALL for s, n, _ in P.PAIRS]); ns = np.array([Z[o][f"W|{s}|{n}|{arm}|n"][0] for o in ALL for s, n, _ in P.PAIRS])
        pu = np.array([Z[o][f"W|{s}|{n}|{arm}|purity"][0] for o in ALL for s, n, _ in P.PAIRS]); ok = ~fb
        for o in ALL:
            for s, n, t in P.PAIRS:
                assert (Z[o][f"W|{s}|{n}|idx"] == t).all()
        D = P.person_scores(CH, ALL, lambda o, s, n: [] if Z[o][f"W|{s}|{n}|{arm}|fb"][0] else [Z[o][f"W|{s}|{n}|{arm}|aE"]])
        g = P.wmean(D, ALL) - cb; lo, hi = P.boot(D - base, ALL)
        sch = [(P.wmean(D[[i for i, o in enumerate(ALL) if P.SCH(o) == k]], [o for o in ALL if P.SCH(o) == k]) - P.wmean(base[[i for i, o in enumerate(ALL) if P.SCH(o) == k]], [o for o in ALL if P.SCH(o) == k])).mean() for k in ("within", "full", "runs")]
        dm = look(G, ns[ok].mean(), 100 * pu[ok].mean())
        print(f"{arm}  {ns[ok].mean():5.0f}  {100 * pu[ok].mean():5.1f}%  {100 * fb.mean():4.1f}% | {dm:+5.2f} | {g.mean():+5.2f} [{lo:+.2f}, {hi:+.2f}] {int((g > 0).sum())}/10 | " + " / ".join(f"{x:+.2f}" for x in sch) + " | " + " ".join(f"{x:+.1f}" for x in g))
    print("\ndose-map values are for true labels with chain-like noise at the arm's mean (n, purity) over non-fallback splits (fallback splits excluded from n and purity).")


if __name__ == "__main__":
    main()
