"""ENSEMBLE CURVE (ledger s269): base size x diversity from stored base outputs (alone, a reading of the base, not a chain score).
Nets: runs/diag/ttt/probs18/ttt18<order>d<draw>_<family>_base-all_loso-s<seed>.npz (proba, index), 4 families x 6 seeds.
Per (order, draw) cell and grid point (f families, s seeds per family): 20 random subsets (fixed rng), mean posterior, person-equal BA.
Intervals (80 %) resample people and (order, draw) cells; the fit BA = a - b/n^c is refitted inside every resample.
Usage: python studies/14nov/ensemble_curve/curve.py [order ...]   (default: within0 within1; draws d1 d2)"""
import itertools, os, sys, warnings
import numpy as np
from scipy.optimize import curve_fit
REPO = os.path.abspath(__file__ + "/../../../.."); sys.path.insert(0, REPO + "/replay")
import env  # noqa: E402
warnings.filterwarnings("ignore")
A = env.A; p0 = "all_loso"; Z = A.L.Z
y, per = Z[f"y|{p0}"].astype(int), Z[f"person|{p0}"]; PEOPLE = np.unique(per)
FAM = ("fb", "fbl", "fbraw", "sinc"); SEEDS = range(6); DRAWS = ("d1", "d2"); ORD = sys.argv[1:] or ["within0", "within1"]
D = REPO + "/runs/diag/ttt/probs18"; NDRAW = 20


def person_ba(P):
    return np.array([np.mean([np.mean(P[(per == q) & (y == c)].argmax(1) == c) for c in range(3)]) for q in PEOPLE]) * 100


def load():
    N, ref = {}, None
    for o, d, f, s in itertools.product(ORD, DRAWS, FAM, SEEDS):
        fn = f"{D}/ttt18{o}{d}_{f}_base-all_loso-s{s}.npz"; assert os.path.exists(fn), fn  # guard 1: every file exists
        z = np.load(fn); ref = z["index"] if ref is None else ref
        assert (z["index"] == ref).all() and len(ref) == len(y) and z["proba"].shape == (len(y), 3), fn  # guard 1b: same rows
        N[o, d, f, s] = z["proba"].astype(np.float64)
    return N


def subset_ba(N, o, d, nets):
    return person_ba(np.mean([N[o, d, f, s] for f, s in nets], 0))


def guard_single(N):  # guard 2: a 1-net subset equals that net's own BA read fresh from its file
    for (o, d, f, s) in N:
        z = np.load(f"{D}/ttt18{o}{d}_{f}_base-all_loso-s{s}.npz"); own = person_ba(z["proba"])
        assert np.allclose(subset_ba(N, o, d, [(f, s)]), own), (o, d, f, s)
    print(f"guard: all {len(N)} single-net subsets equal their own file BA; all files present, index arrays agree")


def grid(N, rng):
    R, size = {}, {}  # (f, s) -> (cells, people)
    for f in range(1, 5):
        for s in range(1, 7):
            rows = []
            for o, d in itertools.product(ORD, DRAWS):
                sub = set(); acc = []
                for _ in range(NDRAW):
                    fams = rng.choice(len(FAM), f, replace=False); nets = [(FAM[i], int(j)) for i in fams for j in rng.choice(6, s, replace=False)]
                    assert len(nets) == f * s and len(set(nets)) == f * s  # guard 3: subset size is what the label says
                    sub.add(tuple(sorted(nets))); acc.append(subset_ba(N, o, d, nets))
                rows.append(np.mean(acc, 0)); size[f, s] = len(sub)
            R[f, s] = np.stack(rows)
    return R, size


def fit(n, b):
    try:
        p, _ = curve_fit(lambda x, a, bb, c: a - bb / x ** c, n, b, p0=(b[-1] + 1, 3, 1), bounds=([0, 0, 0.05], [200, 500, 4]), maxfev=5000); return p
    except Exception:
        return None


def seed_axis(R, pp, oo):
    return np.array([R[4, s][np.ix_(oo, pp)].mean() for s in range(1, 7)])


def stats(R, pp, oo):
    """-> dict of scalars for one (resampled) set of cells/people"""
    g = lambda f, s: R[f, s][np.ix_(oo, pp)].mean(); out = {}
    n = np.array([4 * s for s in range(1, 7)], float); b = seed_axis(R, pp, oo); p = fit(n, b)
    out["seed 1->6"] = g(4, 6) - g(4, 1); out["seed 4->6"] = g(4, 6) - g(4, 4)
    out["fam 1->4 (s=6)"] = g(4, 6) - g(1, 6); out["drop fbraw (3 fam, s=6)"] = np.nan
    if p is not None:
        f = lambda x: p[0] - p[1] / x ** p[2]; out["fit n=24"] = f(24); out["fit 48"] = f(48); out["fit 96"] = f(96)
        out["gain 24->48"] = f(48) - f(24); out["gain 24->96"] = f(96) - f(24); out["c"] = p[2]
    for t in (4, 6, 12):  # family axis at equal total n
        for f, s in ((f, t // f) for f in (1, 2, 3, 4) if t % f == 0 and t // f <= 6):
            out[f"n={t} f={f} s={s}"] = g(f, s)
    return out


def main():
    N = load(); guard_single(N); rng = np.random.default_rng(0); R, size = grid(N, rng)
    assert all(v > 1 for k, v in size.items() if k not in ((4, 6),)), size  # guard 3b: draws differ (only the full set is unique)
    nc = len(ORD) * len(DRAWS); allo, allp = np.arange(nc), np.arange(len(PEOPLE)); pt = stats(R, allp, allo)
    # drop fbraw: 3 families without fbraw, all 6 seeds, against all four (direct, no random draw needed)
    nb = {}
    for k, (o, d) in enumerate(itertools.product(ORD, DRAWS)):
        nb[k] = (subset_ba(N, o, d, [(f, s) for f in FAM if f != "fbraw" for s in SEEDS]), subset_ba(N, o, d, [(f, s) for f in FAM for s in SEEDS]))
    drop = np.stack([nb[k][0] - nb[k][1] for k in range(nc)]); pt["drop fbraw (3 fam, s=6)"] = drop.mean()
    bs = {k: [] for k in pt}; br = np.random.default_rng(1)
    for _ in range(1000):
        pp, oo = br.integers(0, len(PEOPLE), len(PEOPLE)), br.integers(0, nc, nc); st = stats(R, pp, oo); st["drop fbraw (3 fam, s=6)"] = drop[np.ix_(oo, pp)].mean()
        for k in bs:
            if k in st: bs[k].append(st[k])
    iv = {k: np.percentile(v, [10, 90]) if len(v) > 50 else (np.nan, np.nan) for k, v in bs.items()}
    L = [f"s269 ensemble curve; orders {ORD} x draws {DRAWS}; {NDRAW} subsets per cell; {len(PEOPLE)} people; BA in %, person-equal, mean over cells", ""]
    L.append("table: mean BA by f families (rows) x s seeds per family (cols 1..6)")
    for f in range(1, 5): L.append(f"  f={f}  " + " ".join(f"{R[f, s][:, :].mean():6.2f}" for s in range(1, 7)))
    L.append("  (n=f*s nets; distinct subsets per cell: " + ", ".join(f"{k}:{v}" for k, v in size.items() if k[0] == 1 and k[1] == 1) + " ...)")
    L.append(""); L.append("seed axis (f=4), n = 4..24: " + " ".join(f"{v:.2f}" for v in seed_axis(R, allp, allo)))
    L.append("family axis at equal n: " + " | ".join(f"{k} {v:.2f}" for k, v in pt.items() if k.startswith("n=")))
    L.append("fit BA = a - b/n^c on the seed axis (n=4..24), 80 % interval over people and orders (refit per resample):")
    for k in ("seed 1->6", "seed 4->6", "fam 1->4 (s=6)", "drop fbraw (3 fam, s=6)", "fit n=24", "fit 48", "fit 96", "gain 24->48", "gain 24->96", "c"):
        L.append(f"  {k:26s} {pt.get(k, float('nan')):+7.2f}  [{iv[k][0]:+.2f}, {iv[k][1]:+.2f}]")
    o24 = seed_axis(R, allp, allo)[-1]; inside = iv["fit n=24"][0] <= o24 <= iv["fit n=24"][1]
    L.append(f"observed 6-seed (n=24) point {o24:.2f}; inside fit interval at n=24: {inside}")
    g = pt.get("gain 24->96", np.nan); L.append(f"rule: gain 24->96 {g:+.2f} -> " + ("chain check + distillation pre-registration" if g >= 1.0 and inside else "recorded, net-saturated" if g < 0.5 else "undecided"))
    txt = "\n".join(L); print(txt)
    if len(ORD) > 1 or "--smoke" in sys.argv:
        os.makedirs(REPO + "/evidence/ensemble_curve_15nov", exist_ok=True); open(REPO + "/evidence/ensemble_curve_15nov/curve.txt", "w").write(txt + "\n")


if __name__ == "__main__":
    main()
