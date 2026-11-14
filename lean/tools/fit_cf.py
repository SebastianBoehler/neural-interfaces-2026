"""Fit the run-level count filter's parameters on all ten public people (ledger s228c; dev form: the 'lag' detector and
J-wprior/urn of studies/14nov/rundetect, there fitted leave-one-person-out). Writes runs/countfilter/cf_params.npz, the
cf_* arrays lean/tools/build6.py stores as v6_cf_* keys, and
cf_prior.npz (the design prior, read by studies/14nov/cf_reference.py only).

  W, sw    the 6-direction run-discriminant projection of the 246 band log-variances (common.projection on all ten
           people) and its per-direction within-run SD, cross-fitted as in common.projected: the mean over people q of
           q's within-run variance in the projection fitted without q (the fitting people's own scales overfit)
  thr      the detector threshold per check window: common's balanced-error midpoint over every session x order of
           runs/rundetect/ratios.npz ('lag', each session in its own leave-one-person-out projection, so the statistics
           are out of sample as for a new person), all ten people instead of the nine others
  Ls, lp, cap  structured.hyps('wprior') with the design prior counted over all 14 sessions (held = none)

Usage: python lean/tools/fit_cf.py [OUT.npz]"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "studies/14nov/rundetect"))
import common as C  # noqa: E402
import detect as D  # noqa: E402
import jointcount as J  # noqa: E402
import structured as S  # noqa: E402

CHECKS = (40, 50, 60, 80, 100, 150)  # package.py


def main():
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "runs/countfilter/cf_params.npz"
    F = C.blv(); Fc = C.sess_centre(F)
    W, _ = C.projection(F, None)  # all ten people (held = exclude = None)
    ws = []
    for q in C.PEOPLE:  # cross-fitted scale, as common.projected with no held-out person
        Wq, _ = C.projection(F, None, exclude=q); m = C.PER == q; V = Fc[m] @ Wq; r = C.RUN[m]
        M = np.stack([V[r == u].mean(0) for u in r]); ws.append(((V - M) ** 2).mean(0))
    sw = np.sqrt(np.mean(ws, 0))
    R = np.load(ROOT / "runs/rundetect/ratios.npz"); keys = [k for k in R.files if k.startswith("lag|")]
    M = np.stack([R[k] for k in keys]); contig = np.array([D.scheme(k.split("|")[1]) != "full" for k in keys])
    thr = np.array([D.threshold(M[:, D.WS.index(w)], contig) for w in CHECKS])
    Ls, lp, cap = S.hyps("wprior", None)
    capx = np.concatenate([cap, np.full((len(Ls), 2), 12)], 1)  # jointcount.hyp_cache: blocks past the design
    pl, pbh = S.design_prior(None)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, cf_W=W, cf_sw=sw, cf_bands=np.array(C.BANDS, float), cf_fs=np.float64(C.FS), cf_checks=np.array(CHECKS),
             cf_thr=thr, cf_Ls=Ls.astype(np.int64), cf_lp=lp, cf_cap=capx.astype(np.int64), cf_tau=np.float64(J.TAU),
             cf_floor=np.float64(J.FLOOR), cf_K=np.int64(J.K))
    np.savez(out.with_name("cf_prior.npz"), pl=pl, pbh=np.float64(pbh))  # for the reference check (not a package key)
    lopo = np.load(ROOT / "runs/rundetect/decisions.npz")
    med = np.median([lopo[f"lag|{p}"][[list(lopo["W"]).index(w) for w in CHECKS]] for p in C.PEOPLE], 0)
    print(f"wrote {out}: W {W.shape}, sw {np.round(sw, 3)}, thr {np.round(thr, 4)} (leave-one-out median {np.round(med, 4)}), "
          f"P(L=30..36) {np.round(pl, 3)}, P(BH first) {pbh:.3f}, {len(Ls)} hypotheses")


if __name__ == "__main__":
    main()
