"""LABEL-FREE CEILING scorer (ledger s252). Chain = r15 (runs/chainout/s242r15_<order>.npz V6Fr, six bases). Per order,
person: mean over sessions of the mean over the six pairs of the mean over bases of the cell-mean BA of the chain alone or
fused (product) with personal outputs. Personal: labelled (s246: classic.npz aE, ft_*.npz ft200 mean = bE) or pseudo-label
forms hard / soft / filt (cpl_<order>.npz aE; ftpl_*_g<group>.npz mean over families x seeds = bE). Gains against the chain
alone on the same order; 80 % intervals resample people and orders (orders within scheme, schemes weighted equally).
Usage: python studies/14nov/personal_ceiling/score_pl.py > evidence/personal_ceiling_14nov/score_pl.txt"""
import glob, os, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np  # noqa: E402
import splits as S  # noqa: E402
GROUPS = (("within0", "within1", "full0"), ("full1", "runs0", "runs1"))  # = ftnet_pl.GROUPS (that import needs torch)

R = S.REPO; m = S.meta(); SCH = lambda o: o.rstrip("0123456789")  # noqa: E731
ALL = [f"within{i}" for i in range(12)] + [f"full{i}" for i in range(4)] + [f"runs{i}" for i in range(4)]
# top* forms: amendment 3 (cpls_<order>.npz, aE only)
SIX = [o for g in GROUPS for o in g]; FORMS = ("hard", "soft", "filt", "top85", "top70", "top55"); PAIRS = [(s, n, t) for s in S.sessions(m) for n, _, t in S.pairs(m, s)]


def load():
    CH = {o: [np.load(R / f"runs/chainout/s242r15_{o}.npz")[k].astype(np.float64) for k in sorted(np.load(R / f"runs/chainout/s242r15_{o}.npz").files) if k.endswith("|V6Fr")] for o in ALL}
    assert all(len(v) == 6 for v in CH.values()); cl = np.load(S.OUT / "classic.npz"); ft = [np.load(f) for f in sorted(glob.glob(str(S.OUT / "ft_*_s*.npz")))]
    assert len(ft) == 8, len(ft); LAB = {}
    for s, n, t in PAIRS:
        k = f"W|{s}|{n}"; assert (cl[k + "|idx"] == t).all(); LAB[s, n, "aE"] = cl[k + "|aE"]; LAB[s, n, "bE"] = np.mean([f[k + "|ft200"] for f in ft], 0)
    PA = {}
    for o in ALL:
        for f in (S.OUT / f"cpl_{o}.npz", S.OUT / f"cpls_{o}.npz"):
            if f.exists():
                z = np.load(f)
                for s, n, t in PAIRS:
                    for fm in FORMS:
                        if f"W|{s}|{n}|{fm}|aE" in z.files:
                            PA[o, s, n, fm, "aE"] = z[f"W|{s}|{n}|{fm}|aE"]
    FD = Path(os.environ.get("PL_FTPL_DIR", S.OUT))  # amendment 4: primary = runs/personal_ceiling_leo
    if len(glob.glob(str(FD / "ftpl_*_s*_g*.npz"))) < 16:
        sys.exit(f"FATAL: {FD} missing or has fewer than 16 ftpl files")
    print("ftpl dir:", FD)
    for g, os_ in enumerate(GROUPS):
        fs = sorted(glob.glob(str(FD / f"ftpl_*_s*_g{g}.npz")))
        if len(fs) == 8:
            Z = [np.load(f) for f in fs]
            for o in os_:
                for s, n, t in PAIRS:
                    for fm in FORMS[:3]:  # the nets are trained on hard / soft / filt only (top* forms are aE only)
                        PA[o, s, n, fm, "bE"] = np.mean([z[f"W|{s}|{n}|{o}|{fm}"] for z in Z], 0)
        elif fs:
            print(f"group {g}: {len(fs)}/8 net files, skipped")
    return CH, LAB, PA


def person_scores(CH, orders, pers):
    """pers(o, s, n) -> list of personal outputs to fuse with the chain ([] = chain alone) -> (orders, 10)"""
    out = np.zeros((len(orders), 10))
    for i, o in enumerate(orders):
        v = {}
        for s, n, t in PAIRS:
            Q = pers(o, s, n); v.setdefault(s, []).append(np.mean([S.cell_score(m, t, S.fuse(c[t], *Q) if Q else c[t]) for c in CH[o]]))
        out[i] = [np.mean([np.mean(x) for s, x in v.items() if s.startswith(p)]) for p in S.PEOPLE]
    return out


def wmean(D, orders):
    w = np.array([1 / sum(SCH(q) == SCH(o) for q in orders) for o in orders]); return (D * w[:, None]).sum(0) / w.sum()


def boot(D, orders, rng=np.random.default_rng(0)):
    by = {}; [by.setdefault(SCH(o), []).append(i) for i, o in enumerate(orders)]; b = []
    for _ in range(3000):
        pp = rng.integers(0, 10, 10); b.append(np.mean([D[np.array(ix)[rng.integers(0, len(ix), len(ix))]][:, pp].mean() for ix in by.values()]))
    return np.percentile(b, [10, 90])


def main():
    CH, LAB, PA = load()
    for orders, name in ((SIX, "six bE orders"), (ALL, "all 20 orders (aE only)")):
        have = lambda fm, k: all((o, s, n, fm, k) in PA for o in orders for s, n, _ in PAIRS)  # noqa: E731
        base = person_scores(CH, orders, lambda o, s, n: []); rows = {}
        rows["lab c*aE"] = person_scores(CH, orders, lambda o, s, n: [LAB[s, n, "aE"]])
        rows["lab c*bE"] = person_scores(CH, orders, lambda o, s, n: [LAB[s, n, "bE"]])
        rows["lab c*aE*bE"] = person_scores(CH, orders, lambda o, s, n: [LAB[s, n, "aE"], LAB[s, n, "bE"]])
        for fm in FORMS:
            if have(fm, "aE"):
                rows[f"{fm} aE"] = np.array([[np.mean([np.mean([S.cell_score(m, t, PA[o, s, n, fm, "aE"]) for s, n, t in PAIRS if s == ss]) for ss in S.sessions(m) if ss.startswith(p)]) for p in S.PEOPLE] for o in orders])
                rows[f"{fm} c*aE"] = person_scores(CH, orders, lambda o, s, n: [PA[o, s, n, fm, "aE"]])
            if have(fm, "bE"):
                rows[f"{fm} bE"] = np.array([[np.mean([np.mean([S.cell_score(m, t, PA[o, s, n, fm, "bE"]) for s, n, t in PAIRS if s == ss]) for ss in S.sessions(m) if ss.startswith(p)]) for p in S.PEOPLE] for o in orders])
                rows[f"{fm} c*bE"] = person_scores(CH, orders, lambda o, s, n: [PA[o, s, n, fm, "bE"]])
                if have(fm, "aE"):
                    rows[f"{fm} c*aE*bE"] = person_scores(CH, orders, lambda o, s, n: [PA[o, s, n, fm, "aE"], PA[o, s, n, fm, "bE"]])
        cb = wmean(base, orders); lab = wmean(rows["lab c*aE*bE"], orders) - cb
        print(f"\n=== {name}: r15 chain on the test runs, ten-person mean {cb.mean():.2f}")
        print("system          level | gain over chain [80 % people x orders] people up | share of labelled c*aE*bE gain | per person gain P01..P10")
        for k, D in rows.items():
            g = wmean(D, orders) - cb; lo, hi = boot(D - base, orders); lev = wmean(D, orders).mean()
            print(f"  {k:13s} {lev:6.2f} | {g.mean():+5.2f} [{lo:+.2f}, {hi:+.2f}] {int((g > 0).sum()):2d}/10 | {100 * g.mean() / lab.mean():5.0f} % | " + " ".join(f"{x:+.1f}" for x in g))
        print("  chain per person: " + " ".join(f"{x:.1f}" for x in cb))


if __name__ == "__main__":
    main()
