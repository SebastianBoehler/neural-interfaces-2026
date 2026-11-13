"""RETUNE (ledger s166): the stage constants re-chosen under non-chronological delivery (s164: v5.1b's stages lose 1 to
4 points for new people when the session is shuffled). Grid over rebalancing (window W, strength alpha), gated
SessionEM (lambda, n0) and gated TSEM (lambda, n0); each configuration is scored on new people in recording order and
under the three shuffle schemes of shuffle.py (two permutations each), on the exploration bases. Selection is nested
over people (choose on five, score on the other five, both ways) by select_retune.py; fresh bases confirm.
Usage (Leonardo CPU job): JOB=k NJOBS=12 [BASES=B2_pool,B2_s345] python studies/13nov/retune.py   -> runs/night13/retune_<k>.npz"""
import itertools, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("SCHEMES", "full,runs,within"); os.environ.setdefault("PERMS", "2")
import shuffle as S  # noqa: E402  (stores, perm_for, apply; no run on import)
A, SC, gm, L, p0 = S.A, S.SC, S.gm, S.A.L, S.p0
GRID = list(itertools.product((30, 90, 100000), (1.15, 1.5), (0.5, 0.8), (60, 20), (0.15, 0.30, 0.45), (60, 20)))  # rW, ra, eml, emn0, tl, tn0
JOB, NJ = int(os.environ["JOB"]), int(os.environ.get("NJOBS", 12)); BASES = os.environ.get("BASES", "B2_pool,B2_s345").split(",")
MINE = [i for i in range(len(GRID)) if i % NJ == JOB]
print(f"RETUNE job {JOB}/{NJ}: {len(MINE)} of {len(GRID)} configurations, bases {BASES}, schemes rec + {S.SCHEMES} x {S.PERMS}", flush=True)


def pipe(mix, c):
    rW, ra, eml, emn0, tl, tn0 = c; sess = L.Z[f"sess|{p0}"]
    b = L.rebal(mix, sess, alpha=ra, k=5.0, W=rW); g = gm.gate(b, p0)
    F = L.Z[f"GE|{p0}"]; e = np.empty_like(b)
    for s in dict.fromkeys(sess):
        em = gm.EMg(gm.BEM, lam=eml, n0=emn0, m0=20, shrink=0.7, nmin=gm.NMIN)
        for r in np.flatnonzero(sess == s):
            em.g = g[r]; e[r] = em.step(F[r], b[r])
    Lk, tt = L.G.tsem_lik(e, sess, L.Z[f"pc128|{p0}"], rho=0.7); out = e.copy(); m = ~np.isnan(Lk[:, 0])
    w = tl * tt[m] / (tt[m] + tn0) * (1 + gm.BTS * (gm.M.h(tt[m], 30) * g[m])); q = e[m] * Lk[m] ** w[:, None]; out[m] = q / q.sum(1, keepdims=True)
    return out


ORDERS = [("rec", 0)] + [(sc, sd) for sc in S.SCHEMES for sd in range(S.PERMS)]
ident = {p: np.arange(S.N[p]) for p in S.PR}
S.apply(ident)
SHIP = GRID.index((30, 1.15, 0.5, 60, 0.15, 60)); assert SHIP is not None
for bn in BASES:  # the shipped constants must reproduce gm.pipe_g in recording order
    mix = A.mix(bn, p0, A.FAMS)[0]; assert np.allclose(pipe(mix, GRID[SHIP]), gm.pipe_g(mix, p0)), "pipe() with the shipped constants is not gm.pipe_g"
R = {}
for sc, sd in ORDERS:
    S.apply(ident if sc == "rec" else S.perm_for(sc, sd))
    for bn in BASES:
        mix = A.mix(bn, p0, A.FAMS)[0]
        for i in MINE:
            R[f"{i}|{sc}|{sd}|{bn}"] = SC.person(pipe(mix, GRID[i]), p0)
    print(f"  {sc} {sd} done", flush=True)
S.apply(ident)
np.savez(S.REPO + f"/runs/night13/retune_{JOB}.npz", grid=np.array(GRID, float), **R); print("saved", JOB, len(R))
