"""LABEL-FREE CEILING (a) (ledger s252): classic.py's pipelines trained on the r15 chain's pseudo-labels of the four
training runs (never the true labels). Forms per order (plabels_r15.npz): hard = argmax of the raw posterior, unweighted;
soft = the same labels weighted by the max posterior (ttt's wpl); filt = soft on the gated count-filter posterior.
Output runs/personal_ceiling/cpl_<order>.npz: 'W|<session>|<pair>|<form>|<variant>' -> (n, 3), 'W|<session>|<pair>|idx'.
Usage: python studies/14nov/personal_ceiling/classic_pl.py ORDER [ORDER ...] | check"""
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


def main():
    m = S.meta(); F = CL.features(m)
    if sys.argv[1] == "check":
        return check(m, F)
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
