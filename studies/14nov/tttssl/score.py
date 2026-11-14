"""Scorer for TTT-SSL (ledger s255). New people (all_loso), replay score.person (cell BA, mean per person).
Net alone: per seed, mean over seeds; change vs the same form's na (and joint na vs twin na); 80 % intervals resample
people and seeds. Fusion: chain (runs/chainout/s242r15_<order>.npz V6Fr, per base) x net^w (net = mean over seeds),
12 orders x 6 bases; change vs c x na^w (same form) and vs the chain alone; 80 % intervals resample people and orders
(within each scheme; schemes weighted equally). Zero-change check: c x na^w against itself and w = 0 equal to the chain.
Usage: python studies/14nov/tttssl/score.py [PROBS_DIR] > evidence/tttssl_14nov/score.txt"""
import glob, os, re, sys
from pathlib import Path
R = Path(__file__).resolve().parents[3]; sys.path.insert(0, str(R / "replay"))
import numpy as np  # noqa: E402
import score as SC  # noqa: E402

D = Path(sys.argv[1]) if len(sys.argv) > 1 else R / "runs/tttssl/probs"
ORDERS = [f"{s}{i}" for s in ("within", "full", "runs") for i in range(4)]; WS_ = (0.25, 0.5, 1.0); NB = 2000
rng = np.random.default_rng(0)


def person(P):
    return SC.person(P, "all_loso")


def load():
    out = {}
    for f in glob.glob(str(D / "tttssl_*-s*.npz")):
        form, arm, seed = re.match(r"tttssl_(\w+?)_(\w+)-s(\d+)\.npz", os.path.basename(f)).groups()
        z = np.load(f); assert (z["index"] == np.arange(2655)).all(); out.setdefault((form, arm), {})[int(seed)] = z["proba"]
    return out


def boot2(M, nb=NB):
    """M (groups, people) of changes -> 80 % interval of the mean, resampling groups and people"""
    g, p = M.shape; v = [M[rng.integers(0, g, g)][:, rng.integers(0, p, p)].mean() for _ in range(nb)]
    return np.percentile(v, [10, 90])


def alone(P):
    print("## Net alone (new people, mean over seeds; change vs the same form's na [80 % people x seeds] people up)")
    na = {f: {s: person(p) for s, p in P[(f, "na")].items()} for f in ("twin", "C", "P") if (f, "na") in P}
    for f in na:
        print(f"{f:5s} na {np.mean(list(na[f].values())):.2f} ({len(na[f])} seeds)  seed-mean probs {person(np.mean(list(P[(f, 'na')].values()), 0)).mean():.2f}")
    for f in ("C", "P"):
        if f in na and "twin" in na:
            S = sorted(set(na[f]) & set(na["twin"])); M = np.array([na[f][s] - na["twin"][s] for s in S])
            lo, hi = boot2(M); print(f"{f} na - twin na: {M.mean():+.2f} [{lo:+.2f}, {hi:+.2f}] {int((M.mean(0) > 0).sum())}/10")
    for (f, a), ps in sorted(P.items()):
        if a == "na" or f not in na:
            continue
        S = sorted(set(ps) & set(na[f])); M = np.array([person(ps[s]) - na[f][s] for s in S]); lo, hi = boot2(M)
        pp = " ".join(f"{x:+.1f}" for x in M.mean(0))
        print(f"{f:5s} {a:7s} {M.mean():+.2f} [{lo:+.2f}, {hi:+.2f}] {int((M.mean(0) > 0).sum())}/10  seeds {len(S)}  per person {pp}")


def fused(c, q, w):
    F_ = c * np.clip(q, 1e-9, 1) ** w; return F_ / F_.sum(1, keepdims=True)


def fusion(P):
    print("\n## Fusion with the r15 chain (c x net^w, net = seed mean; 12 orders x 6 bases; change [80 % people x orders] people up)")
    CH = {o: [np.load(R / f"runs/chainout/s242r15_{o}.npz")[k].astype(np.float64) for k in sorted(np.load(R / f"runs/chainout/s242r15_{o}.npz").files) if k.endswith("|V6Fr")] for o in ORDERS}
    assert all(len(v) == 6 for v in CH.values())
    base = np.array([np.mean([person(c) for c in CH[o]], 0) for o in ORDERS]); print(f"chain alone {base.mean():.2f}")
    Q = {k: np.mean(list(v.values()), 0) for k, v in P.items()}

    def sc(q, w):
        return np.array([np.mean([person(fused(c, q, w)) for c in CH[o]], 0) for o in ORDERS])
    z0 = sc(next(iter(Q.values())), 0.0); assert np.abs(z0 - base).max() < 1e-12, "w 0 must equal the chain"
    sch = np.array([o.rstrip("0123456789") for o in ORDERS])

    def report(lbl, M):  # M (orders, people)
        per_s = {s: M[sch == s].mean() for s in dict.fromkeys(sch)}; mean = np.mean(list(per_s.values()))
        v = []
        for _ in range(NB):
            pi = rng.integers(0, 10, 10); v.append(np.mean([M[sch == s][rng.integers(0, 4, 4)][:, pi].mean() for s in per_s]))
        lo, hi = np.percentile(v, [10, 90]); ss = " ".join(f"{s} {x:+.2f}" for s, x in per_s.items())
        print(f"{lbl:28s} {mean:+.2f} [{lo:+.2f}, {hi:+.2f}] {int((M.mean(0) > 0).sum())}/10  ({ss})")
    for w in WS_:
        print(f"-- w {w}")
        S = {k: sc(q, w) for k, q in Q.items()}
        for (f, a), M in sorted(S.items()):
            report(f"{f} {a} vs chain", M - base)
            if a != "na" and (f, "na") in S:
                report(f"{f} {a} vs {f} na", M - S[(f, "na")])


if __name__ == "__main__":
    P = load(); print(f"files: {sorted((k, len(v)) for k, v in P.items())}\n")
    alone(P); fusion(P)
