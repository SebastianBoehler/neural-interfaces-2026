"""JOINTCOUNT (ledger s226, step 3): the run-level count filter with the run boundaries as a latent variable. Each
structured hypothesis (structured.py: block lengths from the public design, LOPO prior) carries its own urn filter
(runcount.py's urn: posterior^TAU as likelihood, FLOOR pseudo-items, cap 10 for a 30-window online block, else 12);
the output for window t is the mixture over hypotheses, sum_h w_h * pred_h, with w_h from windows < t. Weights:
  urn   prior x the urn's predictive evidence for the chain's outputs (an exhausted urn is evidence for a boundary)
  both  that x the feature change-point evidence (structured.posterior_path, windows <= t)
Fixed-label mode (labels file, one hypothesis per group) reproduces runcount.py; TRUE runs with true caps validate it.
Causal throughout. Scores per order and arm: change against filter off, new people, person-bootstrap 80 % interval.
Usage: [TAU=2] [FLOOR=0.5] python jointcount.py <mode> order [order ...]
  mode: true | ids:<runids npz>[:cap12] | J:<form>:<urn|both>  (form wcp / wprior / rcp / pos)"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402
import structured as S  # noqa: E402
from scipy.special import logsumexp  # noqa: E402

TAU, FLOOR = float(os.environ.get("TAU", 2)), float(os.environ.get("FLOOR", 0.5)); K = 13; SC = C.A.SC
KK = np.stack(np.meshgrid(np.arange(K), np.arange(K), np.arange(K), indexing="ij"), -1)  # (K, K, K, 3) clipped counts
TRUE_CAP = {u: (len(np.flatnonzero(C.RUN == u)) // 3 if "Online" in u.split("|")[1] else 12) for u in np.unique(C.RUN)}


def urn(P, reset, cap):
    """P (T, 3) chain outputs in delivery order; reset (T, H) bool (block starts), cap (T, H). -> pred (T, H, 3)
    unnormalised (sums = the urn's predictive evidence), exactly runcount.filt per hypothesis (counts clipped at 12, the
    caps are <= 12 so the transition is unchanged)."""
    T, H = reset.shape; al = np.zeros((H, K, K, K)); out = np.zeros((T, H, 3))
    for t in range(T):
        al[reset[t]] = 0; al[reset[t], 0, 0, 0] = 1.0
        L = np.clip(P[t], 1e-9, 1) ** TAU; L /= L.sum()
        tr = np.maximum(cap[t][:, None, None, None, None] - KK[None], 0) + FLOOR; tr /= tr.sum(-1, keepdims=True)
        v = al[..., None] * tr * L; out[t] = v.reshape(H, -1, 3).sum(1); new = np.zeros_like(al)
        new[:, 1:] += v[:, :-1, :, :, 0]; new[:, -1] += v[:, -1, :, :, 0]
        new[:, :, 1:] += v[:, :, :-1, :, 1]; new[:, :, -1] += v[:, :, -1, :, 1]
        new[:, :, :, 1:] += v[:, :, :, :-1, 2]; new[:, :, :, -1] += v[:, :, :, -1, 2]
        al = new / out[t].sum(1)[:, None, None, None]
    return out


def filt_groups(P, rk, lab, capf):
    out = P.copy()
    for u in np.unique(lab):
        r = np.flatnonzero(lab == u); r = r[np.argsort(rk[r])]; c = capf(r)
        pr = urn(P[r], np.eye(len(r), 1, dtype=bool), np.full((len(r), 1), c))[:, 0]; out[r] = pr / pr.sum(1, keepdims=True)
    return out


def hyp_cache(order, form, X, LAM):
    hc = {}
    for s in C.SESSIONS:
        rows = C.order_rows(order, s); p = C.PER[rows[0]]; Ls, lp, cap = S.hyps(form, p)
        post, bounds = S.posterior_path(X[rows], LAM[p], Ls, lp, use_x=True); T = len(rows)
        blk = np.stack([S.block_index(bounds, t) for t in range(T)]); reset = np.vstack([np.ones(len(Ls), bool), blk[1:] != blk[:-1]])
        capx = np.concatenate([cap, np.full((len(Ls), 2), 12)], 1); hc[s] = (rows, lp, post, reset, capx[np.arange(len(Ls))[None], blk])
    return hc


def filt_joint(P, hc, wmode):
    out = P.copy()
    for s, (rows, lp, post, reset, cap) in hc.items():
        pr = urn(P[rows], reset, cap); ev = np.log(np.maximum(pr.sum(2), 1e-300)); cum = np.vstack([np.zeros(len(lp)), np.cumsum(ev, 0)[:-1]])
        lw = (post if wmode == "both" else lp[None]) + cum; w = np.exp(lw - logsumexp(lw, 1, keepdims=True))
        o = (w[..., None] * pr).sum(1); out[rows] = o / o.sum(1, keepdims=True)
    return out


def main():
    mode, orders = sys.argv[1], sys.argv[2:]; rng = np.random.default_rng(0); X, LAM = C.projected() if mode.startswith("J") else (None, None)
    print(f"JOINTCOUNT (s226) {mode}: TAU {TAU}, FLOOR {FLOOR}; filter on - off, offsets off, new people, mean over bases [80 % person bootstrap]")
    for order in orders:
        sc, sd = order.rstrip("0123456789"), int(order[len(order.rstrip("0123456789")):]); g = C.PERMS[f"{sc}|{sd}|{C.p0}"]
        rk = np.empty(C.N, int); rk[g] = np.arange(C.N); z = np.load(f"{C.REPO}/runs/chainout/s215_{order}_off.npz"); bases = sorted({k.split("|")[0] for k in z.files})
        if mode == "true":
            f = lambda P: filt_groups(P, rk, C.RUNI, lambda r: TRUE_CAP[C.RUN[r[0]]])  # noqa: E731
        elif mode.startswith("ids:"):
            parts = mode.split(":"); lab = np.load(C.REPO + "/" + parts[1])[order]; lab = np.char.add(C.SES.astype(str), np.char.add("|", lab.astype(str)))
            f = lambda P: filt_groups(P, rk, lab, lambda r: 12)  # noqa: E731
        else:
            _, form, wm = mode.split(":"); hc = hyp_cache(order, form, X, LAM); f = lambda P: filt_joint(P, hc, wm)  # noqa: E731
        for arm in ("P0r", "V6Fr"):
            d = np.mean([SC.person(f(z[f"{bn}|{arm}"].astype(np.float64)), C.p0) - SC.person(z[f"{bn}|{arm}"].astype(np.float64), C.p0) for bn in bases], 0)
            bt = np.percentile([d[rng.integers(0, 10, 10)].mean() for _ in range(4000)], [10, 90])
            print(f"  {order:8s} {arm:4s} {d.mean():+.2f} [{bt[0]:+.2f}, {bt[1]:+.2f}] | per person " + " ".join(f"{v:+.1f}" for v in d), flush=True)


if __name__ == "__main__":
    main()
