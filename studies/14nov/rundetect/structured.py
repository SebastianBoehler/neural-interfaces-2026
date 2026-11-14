"""RUNDETECT structured boundaries (ledger s226). A hypothesis is a sequence of block lengths taken from the public
design (runs/rundetect/run_design.txt): acquisition Graz 36 and BrainHero L (30-36), then four online runs of 30.
Priors from the OTHER people's design (leave-one-person-out); evidence = the conjugate segment marginal of the windows
delivered so far under the hypothesis' boundaries, in §217's 6-direction view (common.projected). Causal: window t's
label is fixed at t from windows <= t (label = block index with the largest posterior mass at t).
  pos     within: fixed 36 / 36 / 30 x 4, no evidence (the ML expert's position-only form)
  wprior  within: acquisition first, 2 orders x L 30-36, prior only
  wcp     within: the same hypotheses with change-point evidence
  rcp     runs: every placement of the two acquisition blocks among six positions x L 30-36, change-point evidence
Windows past the last hypothesised block form one more block. Writes runs/runids/s226_<form>.npz for every contiguous
order (+ recording) and prints purity / ARI / groups per session per scheme.
Usage: python studies/14nov/rundetect/structured.py"""
import os, sys
from itertools import permutations
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402
from scipy.special import logsumexp  # noqa: E402

LS = np.arange(30, 37); ONLINE = 30; GRAZ = 36
with np.load(C.REPO + "/data/proteus_windows.npz") as _w:
    TASK = _w["task"][C.IDX]


def design_prior(held):
    """P(L) over 30..36 for the BrainHero acquisition run and P(BH acquisition first), from people != held (+0.5 smoothing)."""
    Ls, first = [], []
    for s in C.SESSIONS:
        rows = np.flatnonzero(C.SES == s)
        if C.PER[rows[0]] == held:
            continue
        Ls.append((TASK[rows] == "AcquisitionBH").sum()); first.append(TASK[rows[0]] == "AcquisitionBH")
    pl = np.array([np.sum(np.array(Ls) == v) for v in LS]) + 0.5
    return pl / pl.sum(), (np.sum(first) + 1) / (len(first) + 2)


def hyps(form, held):
    """-> lengths (H, 6), log prior (H,), cap (H, 6): 10 for a 30-window online block, else 12."""
    pl, pbh = design_prior(held); out, lp, cap = [], [], []
    if form == "pos":
        return np.array([[GRAZ, GRAZ] + [ONLINE] * 4]), np.zeros(1), np.array([[12, 12] + [10] * 4])
    for li, L in enumerate(LS):
        if form in ("wprior", "wcp"):
            for bh_first in (False, True):
                out.append([L, GRAZ] + [ONLINE] * 4 if bh_first else [GRAZ, L] + [ONLINE] * 4)
                lp.append(np.log(pl[li]) + np.log(pbh if bh_first else 1 - pbh)); cap.append([12, 12] + [10] * 4)
        elif form == "rcp":
            for g, b in permutations(range(6), 2):  # positions of the Graz and BrainHero acquisition blocks
                seq = [ONLINE] * 6; seq[g], seq[b] = GRAZ, L; c = [10] * 6; c[g] = c[b] = 12
                out.append(seq); lp.append(np.log(pl[li]) - np.log(30)); cap.append(c)
        else:
            raise ValueError(form)
    return np.array(out), np.array(lp), np.array(cap)


def seg_ml(cs, cq, a, b, mu0, lam):
    """log marginal of windows [a, b) (arrays of any shape) as one run: mean ~ N(mu0, lam), noise N(0, 1)."""
    n = (b - a)[..., None].astype(float); s = cs[b] - cs[a]; q = cq[b] - cq[a]; nz = np.maximum(n, 1); xb = s / nz
    v = -0.5 * n * np.log(2 * np.pi) - 0.5 * np.log(1 + n * lam) - 0.5 * ((q - s * xb) + n * (xb - mu0) ** 2 / (1 + n * lam))
    return np.where(n > 0, v, 0.0).sum(-1)


def posterior_path(X, lam, Ls, lp, use_x=True):
    """-> (T, H) log posterior over hypotheses after each window t (windows <= t), and bounds (H, 8)."""
    T = len(X); cs = np.vstack([np.zeros(X.shape[1]), np.cumsum(X, 0)]); cq = np.vstack([np.zeros(X.shape[1]), np.cumsum(X ** 2, 0)])
    bounds = np.concatenate([np.zeros((len(Ls), 1), int), np.cumsum(Ls, 1), np.full((len(Ls), 1), 10 ** 6)], 1)
    out = np.zeros((T, len(Ls)))
    for t in range(T):
        if use_x:
            e = np.minimum(bounds, t + 1); mu0 = cs[t + 1] / (t + 1)
            ll = seg_ml(cs, cq, e[:, :-1], e[:, 1:], mu0, lam).sum(1)
        else:
            ll = 0.0
        w = lp + ll; out[t] = w - logsumexp(w)
    return out, bounds


def block_index(bounds, t):
    return (bounds[:, 1:-1] <= t).sum(1)  # (H,): block of window t under each hypothesis


def labels(X, lam, form, held):
    Ls, lp, _ = hyps(form, held); post, bounds = posterior_path(X, lam, Ls, lp, use_x=form in ("wcp", "rcp"))
    lab = np.zeros(len(X), int)
    for t in range(len(X)):
        k = block_index(bounds, t); lab[t] = np.argmax(np.bincount(k, weights=np.exp(post[t]), minlength=8))
    return lab


FORMS = ("pos", "wprior", "wcp", "rcp"); ORDERS = ["recording"] + [f"{sc}{s}" for sc in ("within", "runs") for s in range(12)]


def main():
    X, LAM = C.projected(); met = {}
    for form in FORMS:
        out = {}
        for o in ORDERS:
            lab = np.full(C.N, -1)
            for s in C.SESSIONS:
                rows = C.order_rows(o, s); p = C.PER[rows[0]]; lab[rows] = labels(X[rows], LAM[p], form, p)
                met.setdefault((form, o.rstrip("0123456789")), []).append(C.purity_ari(lab[rows], C.RUNI[rows]) + (len(np.unique(lab[rows])),))
            assert (lab >= 0).all(); out[o] = lab
        np.savez(C.REPO + f"/runs/runids/s226_{form}.npz", **out)
    print("STRUCTURED (s226): purity / ARI / groups per session vs true runs, mean over sessions x seeds")
    for form in FORMS:
        print(f"  {form:7s} " + " | ".join(f"{sc} {np.mean([m[0] for m in met[form, sc]]):.3f} / {np.mean([m[1] for m in met[form, sc]]):.3f} / {np.mean([m[2] for m in met[form, sc]]):.1f}" for sc in ("recording", "within", "runs")))


if __name__ == "__main__":
    main()
