"""FAIR (ledger s69): re-screen stored members with the chain co-tuned, cross-fitted over people halves (new people only,
no routing). A candidate enters the mix at weight 0.5 / 1 / 2 while rebalancing, SessionEM, the teacher weight and TSEM
are re-chosen; the baseline gets the same search without the candidate.
Usage: NJ=6 python replay/fair.py BASE [name-filter]   (candidates: see `candidates()`; `null_*` duplicate a shipped member)"""
import glob, itertools, os, sys
import numpy as np
import grid as G
import score as SC
from multiprocessing import Pool
REPO = os.path.abspath(__file__ + "/../.."); p = "all_loso"; Z = G.Z; N = len(Z[f"y|{p}"]); WX = tuple(float(v) for v in os.environ.get("FAIR_W", "0.5,1,2").split(","))
CFG = list(itertools.product(G.ALPHAS, G.EMLAMS, (2.0,), G.WTS)); SHIP = CFG.index(G.SHIP); TL = G.TLAMS; L15 = TL.index(0.15)
G1 = [G.PEOPLE.index(q) for q in SC.G1]; G2 = [G.PEOPLE.index(q) for q in SC.G2]


def candidates(base):
    out = {}
    for d in ("sbnx/probs", "netaug/probs", "ocunet/probs", "rest4/probs", "rest4/as_base"):
        fs = sorted(glob.glob(f"{REPO}/runs/{d}/*-{p}-s*.npz"))
        for name in sorted({os.path.basename(f).rsplit(f"-{p}-", 1)[0] for f in fs}):
            zs = [np.load(f) for f in fs if os.path.basename(f).startswith(name + f"-{p}-")]
            assert zs and all((z["index"] == np.arange(N)).all() for z in zs), name
            out[("asb_" if d.endswith("as_base") else "") + name] = np.mean([z["proba"] for z in zs], 0)
    for tag, f in (("aec", "aec/out"), ("cfc", "aec/out_cfc")):
        z = np.load(f"{REPO}/runs/{f}/car_fb7_full_t0.1_le_k2.npz"); assert (z[f"index|{p}"] == np.arange(N)).all()
        out[tag + "_lr3"], out[tag + "_lr10"] = z[f"lr3|{p}"], z[f"lr10|{p}"]
    Ps = Z[f"{base}|Ps|{p}"]
    out.update({"null_core": Ps[0], "null_shallow": Ps[3], "null_sbnfb": Ps[5], "null_sbnsinc": Ps[7]})
    assert all(v.shape == (N, 3) and np.isfinite(v).all() for v in out.values())
    return out


def task(a):
    base, ci, w, E = a; al, le, wsl, wt = CFG[ci]
    Ps, ws = Z[f"{base}|Ps|{p}"], Z[f"{base}|ws|{p}"].copy(); ws[0], ws[-1] = wsl, wt
    mix = (sum(x * P for x, P in zip(ws, Ps)) + (w * E if w else 0.0)) / (ws.sum() + w)
    e = G.SE.apply(Z[f"GE|{p}"], G.rebal(mix, Z[f"sess|{p}"], al), Z[f"sess|{p}"], lam=le, n0=60, m0=20, shrink=0.7)
    L, tt = G.tsem_lik(e, Z[f"sess|{p}"], Z[f"pc128|{p}"], rho=0.7)
    return ci, w, np.stack([G.scores(G.apply_lam(e, L, tt, lt), p) for lt in TL])


def run(pool, base, E, weights):
    R = np.zeros((len(weights), len(CFG), len(TL), 10))
    for ci, w, s in pool.imap_unordered(task, [(base, ci, w, E) for ci in range(len(CFG)) for w in weights]):
        R[weights.index(w), ci] = s
    return R


def crossfit(R):  # R (..., 10) flattened over the search axes -> mean of the two held-out halves
    F = R.reshape(-1, 10); a = F[F[:, G1].mean(1).argmax()][G2].mean(); b = F[F[:, G2].mean(1).argmax()][G1].mean(); return 0.5 * (a + b)


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "1"); base = sys.argv[1]; flt = sys.argv[2] if len(sys.argv) > 2 else ""
    C = {k: v for k, v in candidates(base).items() if flt in k}; rng = np.random.default_rng(0)
    with Pool(int(os.environ.get("NJ", "6"))) as pool:
        B = run(pool, base, None, [0.0])[0]; ship = B[SHIP, L15]
        cached = G.scores(np.load(G.W + "/post_cache.npz")[f"{base}|tsem|{p}"], p)
        assert np.abs(ship - cached).max() < 1e-6, f"shipped setting does not reproduce the cached chain: {np.abs(ship - cached).max()}"
        nb = crossfit(B); print(f"{base}: {len(C)} candidates, {len(CFG)} settings x {len(TL)} lambdas | baseline shipped {ship.mean():.2f}, best of grid {B.reshape(-1, 10).mean(1).max():.2f}, cross-fit {nb:.2f}", flush=True)
        print("candidate              alone | old screen (shipped, w 1) | weight only, cross-fit | co-tuned in-sample | co-tuned cross-fit | P04 at old screen")
        for name, E in C.items():
            R = run(pool, base, E, list(WX)); alone = G.scores(E, p).mean(); old = R[WX.index(1.0), SHIP, L15] - ship
            RW = np.concatenate([B[SHIP, L15][None], R[:, SHIP, L15]])  # weight 0 / 0.5 / 1 / 2 at shipped constants
            semi = crossfit(RW) - 0.5 * (ship[G1].mean() + ship[G2].mean()); allR = np.concatenate([B[None], R])
            ins = allR.reshape(-1, 10).mean(1).max() - B.reshape(-1, 10).mean(1).max(); nest = crossfit(allR) - nb
            bt = np.percentile([old[rng.integers(0, 10, 10)].mean() for _ in range(2000)], [10, 90])
            print(f"{name:22s} {alone:5.1f} | {old.mean():+.2f} [{bt[0]:+.2f}, {bt[1]:+.2f}] | {semi:+.2f} | {ins:+.2f} | {nest:+.2f} | {old[3]:+.1f}", flush=True)
