"""NOISE (ledger s83): in-session own-space learning on chain labels, true labels, and true labels carrying the chain's
error counts at random places (same selected windows). New people. Usage: python studies/12nov/ownnoise.py"""
import os, sys
import numpy as np
from joblib import Parallel, delayed
F = os.path.abspath(__file__ + "/../ownttt.py"); exec(compile(open(F).read().split("\nfor bn in os.environ")[0], F, "exec"))  # noqa: S102


def session2(g, P, ytrue, mode, seed):
    n = len(g); H = {k: np.full((n, 3), np.nan) for k in "TCS"}; pl_, cf = P.argmax(1), P.max(1); rng = np.random.default_rng(seed); nerr = []
    for t in range(T0, n, STEP):
        q = max(4, int(RHO * t / 3)); sel = np.sort(np.concatenate([(ik := np.flatnonzero(pl_[:t] == k))[np.argsort(-cf[ik])[:q]] for k in range(3)]))
        if mode == "true":
            lab = ytrue.copy()
        elif mode == "chain":
            lab = pl_.copy()
        else:  # the chain's errors on sel, moved to random windows of the same true class within sel
            lab = ytrue.copy(); wrong = sel[pl_[sel] != ytrue[sel]]
            for k in range(3):
                pool = sel[ytrue[sel] == k]; errs = wrong[ytrue[wrong] == k]
                if len(errs):
                    at = rng.choice(pool, len(errs), replace=False); lab[at] = pl_[errs]
            assert np.sum(lab[sel] != ytrue[sel]) == len(wrong)
        nerr.append(np.mean(lab[sel] != ytrue[sel]))
        if min(np.sum(lab[sel] == k) for k in range(3)) < 4:
            continue
        o = fit_block(g, t, lab, sel)
        for k in "TCS":
            H[k][t:t + STEP] = o[k]
    return H, float(np.mean(nerr)) if nerr else np.nan


p = "all_loso"; rng0 = np.random.default_rng(0)
for bn in ("B2_pool", "B2_s345"):
    base = A.system(bn, A.FAMS)["out"]; ss, idx, y = Z[f"sess|{p}"], Z[f"idx|{p}"], Z[f"y|{p}"]; U = np.unique(ss); ref = SC.person(base[p], p)
    print(f"{bn}: chain new {ref.mean():.2f}", flush=True)
    for mode in ("chain", "moved", "true"):
        res = Parallel(n_jobs=6)(delayed(session2)(idx[ss == s], base[p][ss == s], y[ss == s], mode, i) for i, s in enumerate(U))
        HH = {k: np.full((len(y), 3), np.nan) for k in "TCS"}; TT = np.zeros(len(y))
        for s, (h, _) in zip(U, res):
            r = np.flatnonzero(ss == s); TT[r] = np.arange(len(r))
            for k in "TCS":
                HH[k][r] = h[k]
        ok = ~np.isnan(HH["T"][:, 0]); acc = 100 * np.mean(sum(np.log(np.clip(HH[k][ok], 1e-3, 1)) for k in "TCS").argmax(1) == y[ok])
        print(f"  {mode:6s}: label error on the selected windows {100 * np.nanmean([e for _, e in res]):.1f}% | learner alone {acc:.1f}%", flush=True)
        for lam in (0.25, 0.5):
            out = fuse(base[p], HH, TT, lam, "TCS"); d = SC.person(out, p) - ref; bt = np.percentile([d[rng0.integers(0, 10, 10)].mean() for _ in range(4000)], [10, 90])
            print(f"     lambda {lam}: new {d.mean():+.2f} [{bt[0]:+.2f}, {bt[1]:+.2f}] | P04 {d[3]:+.1f} | " + " ".join(f"{v:+.1f}" for v in d), flush=True)
