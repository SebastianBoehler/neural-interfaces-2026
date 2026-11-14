"""CAUSAL PERSONALISATION refits (ledger s253). For one order and base of the r15 chain, per session in delivery order: from
window T0 every K0 windows, fast.fit_predict (log-Euclidean l1, FBCSP a2, slow a4, mean lE) on the session's windows < t (labels = the chain's emitted outputs for those
windows; forms hard / soft / filt) predicts windows t .. t+KMAX-1. For each k in KS (multiples of K0, so refit points nest)
window i takes the prediction of refit point T0 + k * floor((i - T0) / k). Inputs runs/causalpers/chain.npz (chainf.py) and
feat.npz (feats.py, per-window features). Output runs/causalpers/pp_<order>_b<base>.npz '<form>|<pipe>|k<k>' -> (N, 3),
NaN before T0. FORMS=<list of SEL forms> -> pp_<order>_b<base>_<forms joined by _>.npz: hard raw labels, unweighted refit
on a selected prefix (top: confidence >= prefix median; cqQ: per chain class the top Q % most confident; agree: raw and
count-filter argmax agree; agcq: cq50 inside agree); 'nfail|<form>' counts refits with < 2 label classes (prediction uniform). Labels never see a window >= t.
Usage: python studies/14nov/causalpers/run.py ORDER BASE [BASE ...] | smoke"""
import os, sys, time
from pathlib import Path
HERE = Path(__file__).resolve().parent; REPO = HERE.parents[2]; sys.path[:0] = [str(HERE), str(REPO / "studies/14nov/personal_ceiling")]
import numpy as np  # noqa: E402
import fast as FA  # noqa: E402
import feats as FT  # noqa: E402
import splits as S  # noqa: E402

T0, K0, KS = 20, 10, (10, 20); KMAX = max(KS)  # s253 amendment: fast pipelines, k 5 dropped
PIPES = tuple(os.environ.get("PIPES", ",".join(FA.PIPES)).split(",")); assert set(PIPES) <= set(FA.PIPES), PIPES  # s260: PIPES=a2 computes only a2
DRAW = os.environ.get("DRAW", ""); assert DRAW in ("", "d1", "d2"), DRAW  # s260: fresh r15 draws; own chain file, own pp files
FORMS = tuple(os.environ.get("FORMS", "hard,soft,filt").split(",")); SFX = "" if FORMS == ("hard", "soft", "filt") else "_" + "_".join(FORMS)
if PIPES != FA.PIPES:
    SFX += "_p" + "".join(PIPES)  # pipelines subset: file name says so, never read as a full file
SFX += "_" + DRAW if DRAW else ""
SEL = ("top", "cq50", "cq25", "agree", "agcq")  # selection forms: hard raw labels, unweighted refit on a selected prefix
assert set(FORMS) <= {"hard", "soft", "filt", *SEL}, FORMS
assert all(k % K0 == 0 for k in KS)


def labels(Z, order, b, form):
    """-> (labels, weights or None, selection fn tr -> tr subset or None)"""
    P = Z[f"{order}|{b}|{'filt' if form == 'filt' else 'raw'}"]; y = P.argmax(1)
    if form not in SEL:
        return y, (None if form == "hard" else P.max(1)), None
    c, ag = P.max(1), y == Z[f"{order}|{b}|filt"].argmax(1)

    def cq(tr, q):  # per chain class, the top-q most confident windows of the prefix (at least 3), as TSEM picks templates
        keep = []
        for k in np.unique(y[tr]):
            r = tr[y[tr] == k]; keep.append(r[np.argsort(-c[r])[:max(3, int(np.ceil(q * len(r))))]])
        return np.sort(np.concatenate(keep))
    sel = {"top": lambda tr: tr[c[tr] >= np.median(c[tr])], "cq50": lambda tr: cq(tr, 0.5), "cq25": lambda tr: cq(tr, 0.25),
           "agree": lambda tr: tr[ag[tr]], "agcq": lambda tr: cq(tr[ag[tr]], 0.5)}[form]
    return y, None, sel


def session_refits(F, T, y, w, sel, rows, out, form, nfail, tmax=None):
    """rows: one session's global rows in delivery order; fills out[(form, pipe, k)][rows[i]] for i >= T0"""
    n = len(rows) if tmax is None else min(len(rows), tmax)
    for t in range(T0, n, K0):
        tr, te = rows[:t], rows[t:t + KMAX]
        if sel is not None:
            tr = sel(tr)
        if len(np.unique(y[tr])) < 2:
            nfail[form] += 1; pr = {p: np.full((len(te), 3), 1 / 3) for p in PIPES}
        else:
            pr = FA.fit_predict(F, T, y, tr, te, None if w is None else w[tr], PIPES)
        for k in KS:
            if (t - T0) % k == 0:
                for p in PIPES:
                    out[form, p, k][te[:k]] = pr[p][:k]


def run(order, b, Z, F, T, tmax=None, sessions=None):
    N = len(Z["y"]); out = {(f, p, k): np.full((N, 3), np.nan) for f in FORMS for p in PIPES for k in KS}; nfail = dict.fromkeys(FORMS, 0)
    dl, sess = Z[f"{order}|deliv"], Z["sname"]
    for form in FORMS:
        y, w, sel = labels(Z, order, b, form)
        for s in (dict.fromkeys(sess[dl]) if sessions is None else sessions):
            rows = dl[sess[dl] == s]; session_refits(F, T, y, w, sel, rows, out, form, nfail, tmax)
    return out, nfail


def main():
    Z = dict(np.load(REPO / f"runs/causalpers/chain{DRAW}.npz")); assert str(Z["draw"]) == DRAW, ("chain file draw", str(Z["draw"]), DRAW); F = FT.load(); T = FA.tsle(F); Z["sname"] = S.meta()["sess"]
    assert all(len(set(Z["sess"][Z["sname"] == s])) == 1 for s in set(Z["sname"])) and len(set(Z["sname"])) == len(set(Z["sess"])) == 14
    if sys.argv[1] == "smoke":  # two sessions, first 60 windows: timing, layout and causality check
        ss = ["P01_1", "P04_2"]; t0 = time.time(); out, nf = run("full0", 0, Z, F, T, tmax=60, sessions=ss); dt = time.time() - t0
        n = 3 * 2 * len(range(T0, 60, K0)); print(f"smoke {dt:.1f}s, {n} refits, {1000 * dt / n:.0f} ms per fit_predict (all pipelines) nfail {nf}")
        pos = Z["full0|pos"]; v = ~np.isnan(out["soft", "lE", 10][:, 0])
        assert v.sum() == 2 * 40 and (pos[v] >= T0).all() and (pos[v] < 60).all() and np.allclose(out["soft", "lE", 10][v].sum(1), 1)
        Z2 = dict(Z); P = Z["full0|0|raw"].copy(); late = pos >= 40; P[late] = P[late][:, ::-1]; Z2["full0|0|raw"] = P
        o2, _ = run("full0", 0, Z2, F, T, tmax=60, sessions=ss); early = v & (pos < 40); lt = v & (pos >= 45)
        for k in KS:
            assert np.array_equal(o2["hard", "lE", k][early], out["hard", "lE", k][early]), k  # labels at >= 40 unseen before 40
        assert not np.allclose(o2["hard", "lE", 10][lt], out["hard", "lE", 10][lt]), "perturbed labels had no effect"
        print("SMOKE OK"); return
    order = sys.argv[1]
    for b in map(int, sys.argv[2:]):
        f = REPO / f"runs/causalpers/pp_{order}_b{b}{SFX}.npz"
        if f.exists():
            print("exists", f); continue
        t0 = time.time(); out, nf = run(order, b, Z, F, T)
        np.savez(f, **{f"{fm}|{p}|k{k}": v for (fm, p, k), v in out.items()}, **{f"nfail|{k}": v for k, v in nf.items()})
        print(f"{order} b{b} {time.time() - t0:.0f}s nfail {nf} -> {f}", flush=True)


if __name__ == "__main__":
    main()
