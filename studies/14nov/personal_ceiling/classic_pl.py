"""LABEL-FREE CEILING (a) (ledger s252): classic.py's pipelines trained on the r15 chain's pseudo-labels of the four
training runs (never the true labels). Forms per order (plabels_r15.npz): hard = argmax of the raw posterior, unweighted;
soft = the same labels weighted by the max posterior (ttt's wpl); filt = soft on the gated count-filter posterior.
Output runs/personal_ceiling/cpl_<order>.npz: 'W|<session>|<pair>|<form>|<variant>' -> (n, 3), 'W|<session>|<pair>|idx'.
Usage: python studies/14nov/personal_ceiling/classic_pl.py ORDER [ORDER ...] | check | randnoise | sel ORDER [ORDER ...]"""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np  # noqa: E402
import classic as CL  # noqa: E402
import splits as S  # noqa: E402

FORMS = ("hard", "soft", "filt")


def labels(PL, order, form):
    P = PL[f"{order}|{'filt' if form == 'filt' else 'raw'}"]; return P.argmax(1), (None if form == "hard" else P.max(1))


def check(m, F):
    """zero-change: true labels with w=None reproduce classic.npz (s246) on two sessions"""
    ref = np.load(S.OUT / "classic.npz")
    for s in ("P01_1", "P04_2"):
        for name, trn, tst in S.pairs(m, s)[:2]:
            got = CL.fit_predict(F, m["y"], trn, tst)
            for k in ("a1", "a2", "a3", "a4", "aE"):
                d = np.abs(got[k] - ref[f"W|{s}|{name}|{k}"]).max(); assert d < 1e-8, (s, name, k, d)
    print("CHECK OK: true labels reproduce classic.npz", flush=True)


def randnoise(m, F, order="within0", seed=0):
    """diagnostic: true labels with the chain's per-session error rate (order's hard labels) flipped at random to another
    class -> is aE's drop on pseudo-labels the noise rate or the chain's systematic errors?"""
    PL = np.load(S.OUT / "plabels_r15.npz"); yc = PL[f"{order}|raw"].argmax(1); y = m["y"]; rng = np.random.default_rng(seed); yr = y.copy()
    for s in S.sessions(m):
        r = np.flatnonzero(m["sess"] == s); k = int(round(np.mean(yc[r] != y[r]) * len(r))); f = rng.choice(r, k, replace=False); yr[f] = (y[f] + rng.integers(1, 3, k)) % 3
    sc = {"rand": [], "chain": []}
    for s in S.sessions(m):
        for name, trn, tst in S.pairs(m, s):
            sc["rand"].append(S.cell_score(m, tst, CL.fit_predict(F, yr, trn, tst)["aE"])); sc["chain"].append(S.cell_score(m, tst, CL.fit_predict(F, yc, trn, tst)["aE"]))
    print(f"label acc rand {np.mean(yr == y):.3f} chain {np.mean(yc == y):.3f} | aE mean over pairs: random noise {np.mean(sc['rand']):.1f}, chain labels {np.mean(sc['chain']):.1f}", flush=True)


SEL = (85, 70, 55)


def selected(m, F, order):
    """s252 amendment 3: aE on the top f % of the training runs' windows by the chain's max posterior, chain argmax labels"""
    f = S.OUT / f"cpls_{order}.npz"
    if f.exists():
        return print("exists", f)
    P = np.load(S.OUT / "plabels_r15.npz")[f"{order}|raw"]; yp, conf = P.argmax(1), P.max(1); res = {}; t0 = time.time()
    for s in S.sessions(m):
        for name, trn, tst in S.pairs(m, s):
            res[f"W|{s}|{name}|idx"] = tst; rk = trn[np.argsort(-conf[trn], kind="stable")]
            for q in SEL:
                sub = np.sort(rk[:int(round(q / 100 * len(trn)))])
                res[f"W|{s}|{name}|top{q}|aE"] = CL.fit_predict(F, yp, sub, tst)["aE"]
                res[f"W|{s}|{name}|top{q}|purity"] = np.array([np.mean(yp[sub] == m["y"][sub])])
    np.savez(f, **res); print("saved", f, f"{time.time() - t0:.0f}s", flush=True)


def main():
    m = S.meta(); F = CL.features(m)
    if sys.argv[1] == "sel":
        return [selected(m, F, o) for o in sys.argv[2:]]
    if sys.argv[1] == "check":
        return check(m, F)
    if sys.argv[1] == "randnoise":
        return randnoise(m, F)
    PL = np.load(S.OUT / "plabels_r15.npz")
    for order in sys.argv[1:]:
        t0 = time.time(); res = {}; f = S.OUT / f"cpl_{order}.npz"
        if f.exists():
            print("exists", f); continue
        for form in FORMS:
            yp, w = labels(PL, order, form)
            for s in S.sessions(m):
                for name, trn, tst in S.pairs(m, s):
                    res[f"W|{s}|{name}|idx"] = tst
                    for k, v in CL.fit_predict(F, yp, trn, tst, None if w is None else w[trn]).items():
                        res[f"W|{s}|{name}|{form}|{k}"] = v
            acc = np.mean([S.cell_score(m, res[f"W|{s}|{n}|idx"], res[f"W|{s}|{n}|{form}|aE"]) for s in S.sessions(m) for n, _, _ in S.pairs(m, s)])
            print(f"{order} {form} aE mean {acc:.1f} {time.time() - t0:.0f}s", flush=True)
        np.savez(f, **res); print("saved", f, flush=True)


if __name__ == "__main__":
    main()
