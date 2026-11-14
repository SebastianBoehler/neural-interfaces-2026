"""v6.2c reference check: the packaged count filter (lean/src62c/countfilter.py) against ni26-push's dev code
(studies/14nov/rundetect at 3a10ba8, pinned in studies/14nov/cf_ref: detect.stat 'lag', structured.hyps 'wprior' / posterior_path bounds / block_index,
jointcount.urn and its J-wprior/urn mixture as in filt_joint, package.py's latch and gate), window by window.
Both use the ten-person fit (lean/tools/fit_cf.py): the dev functions get the package's W, sw and thresholds and the
ten-person design prior, so any difference is the packaging. Recognition off (both gates forced off): the filter's input
is the package's own unfiltered output, streamed in the delivery order.
Usage: python studies/14nov/cf_reference.py PKG [--order SCHEME:SEED:PERMS.npz] [--only K] [--raise-at N]
  --raise-at N: the filter's urn raises at window N; asserts errors 1 and the unfiltered output from N on
Prints one JSON line per session and a RESULT line; exit 1 on any difference beyond 1e-12 or a different latch window."""
import copy, json, sys, time, types
from pathlib import Path
import numpy as np
import torch
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "experiments"), str(ROOT / "lean/tools")]
import protocol  # noqa: E402
from record import load_model  # noqa: E402
# the dev modules import 'common', which loads the whole replay environment; these functions read only the stand-ins
sys.modules["common"] = types.SimpleNamespace(REPO=str(ROOT), IDX=slice(None), A=types.SimpleNamespace(SC=None), RUN=np.array([], str),
                                              p0="all_loso")
sys.path.insert(0, str(ROOT / "studies/14nov/cf_ref"))  # the dev modules pinned at 3a10ba8
import detect as D  # noqa: E402
import jointcount as J  # noqa: E402
import structured as S  # noqa: E402
from scipy.signal import butter, sosfiltfilt  # noqa: E402

arg = lambda k, d: type(d)(sys.argv[sys.argv.index(k) + 1]) if k in sys.argv else d  # noqa: E731
BANDS, FS = [(4, 8), (8, 13), (13, 20), (20, 30), (30, 45), (45, 58)], 120.0  # common.BANDS / FS
PRIOR = np.load(ROOT / "runs/countfilter/cf_prior.npz")
S.design_prior = lambda held: (PRIOR["pl"], float(PRIOR["pbh"]))  # the ten-person design prior


def blv(X):  # common.blv's expression on the delivered windows, cached as float32 there
    F = np.concatenate([np.log(sosfiltfilt(butter(4, b, "bandpass", fs=FS, output="sos"), X.astype(np.float64), axis=-1).var(-1) + 1e-12) for b in BANDS], 1)
    return F.astype(np.float32)


def reference(X, P, z):
    """dev: latch (package.latch), then J-wprior/urn (jointcount.hyp_cache + filt_joint) gated from the latch window."""
    T = len(X); Xv = (blv(X) @ z["v6_cf_W"]) / z["v6_cf_sw"]; checks = [int(w) for w in z["v6_cf_checks"]]
    thr = dict(zip(checks, z["v6_cf_thr"])); ws = [w for w in checks if w <= T]; v = D.stat(Xv, np.arange(T), "lag", ws)
    hit = [w for w, x in zip(ws, v) if x < thr[w]]; latch = hit[0] if hit else None
    Ls, lp, cap = S.hyps("wprior", None)
    assert np.array_equal(Ls, z["v6_cf_Ls"]) and np.allclose(lp, z["v6_cf_lp"], rtol=0, atol=0) and np.array_equal(
        np.concatenate([cap, np.full((len(Ls), 2), 12)], 1), z["v6_cf_cap"]), "packaged hypotheses differ from structured.hyps"
    _, bounds = S.posterior_path(Xv, None, Ls, lp, use_x=False)  # urn mode reads only the bounds (jointcount.hyp_cache)
    blk = np.stack([S.block_index(bounds, t) for t in range(T)]); reset = np.vstack([np.ones(len(Ls), bool), blk[1:] != blk[:-1]])
    capx = np.concatenate([cap, np.full((len(Ls), 2), 12)], 1)
    F = J.filt_joint(P, {"s": (np.arange(T), lp, None, reset, capx[np.arange(len(Ls))[None], blk])}, "urn")
    out = P.copy()
    if latch is not None:
        out[latch:] = F[latch:]
    return out, latch


def main():
    pkg = Path(sys.argv[1]); d = protocol.load(); torch.set_num_threads(1); z = np.load(pkg / "weights.npz")
    assert int(z["v6_cf_on"]) == 1, "the count filter is off in this package"
    order, ra = arg("--order", "recording"), arg("--raise-at", -1)
    ss = list(dict.fromkeys(d["stream"])); ss = [ss[arg("--only", 0)]] if "--only" in sys.argv else ss
    g = None
    if order != "recording":
        sc, sd, f = order.split(":"); g = np.load(f)[f"{sc}|{sd}|all_loso"]
    m = load_model(pkg, None, "cf"); ok, worst = True, 0.0
    for s in ss:
        rows = g[d["stream"][g] == s] if g is not None else np.flatnonzero(d["stream"] == s)
        X = np.nan_to_num(np.asarray(d["X"][rows], np.float32)); c = copy.deepcopy(m)
        for r in (c.route, getattr(c, "routeB", None)):
            if r is not None:
                r.contrast.tau = r.identity.tau = np.inf
        c.reset_state(); P, Q, sec = [], [], []
        if ra >= 0:
            orig = c.cf.urn
            c.cf.urn = lambda p, o=orig, cf=c.cf: (_ for _ in ()).throw(RuntimeError("forced test exception")) if cf.t == ra else o(p)
        for x in X:
            p = c.step(x); t0 = time.perf_counter(); q = c.cf.step(x, p); sec.append(time.perf_counter() - t0)
            P.append(np.asarray(p, np.float64)); Q.append(np.asarray(q, np.float64))
        P, Q = np.array(P), np.array(Q); R, latch = reference(X, P, z)
        if ra >= 0:  # from the error on: the unfiltered output; before it: the dev reference
            good = c.cf.log["err"] == 1 and np.array_equal(Q[ra:], P[ra:]) and float(np.abs(Q[:ra] - R[:ra]).max()) <= 1e-12
            diff = float(np.abs(Q[:ra] - R[:ra]).max())
        else:
            diff = float(np.abs(Q - R).max()); good = diff <= 1e-12 and c.cf.latch == latch and c.cf.log["err"] == 0
        ok &= bool(good); worst = max(worst, diff) if np.isfinite(diff) else float("inf")
        print(json.dumps(dict(session=s.split("/")[-1], windows=len(rows), latch_pkg=c.cf.latch, latch_dev=latch, max_diff=diff,
                              bitwise=bool(np.array_equal(Q, R)), labels_changed=int((Q.argmax(1) != P.argmax(1)).sum()),
                              labels_differ_dev=int((Q.argmax(1) != R.argmax(1)).sum()), err=c.cf.log["err"],
                              ms_per_window=round(1e3 * float(np.mean(sec)), 3), ok=bool(good))), flush=True)
    print("RESULT", json.dumps(dict(package=str(pkg), order=order, raise_at=ra, max_diff=worst, ok=bool(ok))), flush=True)
    assert ok, "count-filter reference check failed"


if __name__ == "__main__":
    main()
