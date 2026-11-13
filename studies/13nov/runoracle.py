"""RUNORACLE (ledger s168): how much of the shuffle loss (s164) is the stages mixing runs? ORACLE (never shippable):
the stages get the true run of every delivered window, so each run is processed as its own sub-session (rebalancing,
SessionEM, TSEM templates from the run's delivered windows only). Scored for new people in recording order and under
full / runs / within (two permutations each), six bases, against the shipped stages under the same order. If the
oracle recovers most of the loss, latent-run forms (clustering delivered windows on slow EEG state) are worth building.
No cue-order or label information is used; the run id is the oracle's only extra input.
Usage: python studies/13nov/runoracle.py"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("SCHEMES", "full,runs,within"); os.environ.setdefault("PERMS", "2")
import shuffle as S  # noqa: E402
A, SC, gm, p0 = S.A, S.SC, S.gm, S.p0
iB = [i for i, d in enumerate(S.DICTS) if d is A.pl.B][0]; IDX = S.ORIG[iB][f"idx|{p0}"].astype(int)
SESS0 = {i: o[f"sess|{p0}"].copy() for i, o in enumerate(S.ORIG) if f"sess|{p0}" in o}
SESSRUN = {i: np.char.add(np.char.add(v.astype(str), "#"), S.RUN[IDX]) for i, v in SESS0.items()}
n_sub = len(np.unique(next(iter(SESSRUN.values())))); n_sess = len(np.unique(next(iter(SESS0.values()))))
print(f"RUNORACLE: {n_sess} sessions split into {n_sub} run sub-sessions (oracle)", flush=True)


def run(P, oracle):
    for i in SESS0:  # the session labels the stages see; permutations are always made from the true sessions first
        S.ORIG[i][f"sess|{p0}"] = SESSRUN[i] if oracle else SESS0[i]
    S.apply(P); out = {bn: SC.person(gm.pipe_g(A.mix(bn, p0, A.FAMS)[0], p0), p0) for bn in S.BASES}
    for i in SESS0:
        S.ORIG[i][f"sess|{p0}"] = SESS0[i]
    return out


ident = {p: np.arange(S.N[p]) for p in S.PR}; ORD = [("rec", 0)] + [(sc, sd) for sc in S.SCHEMES for sd in range(S.PERMS)]
REF = run(ident, False); R = {}
for sc, sd in ORD:
    P = ident if sc == "rec" else S.perm_for(sc, sd)
    R[sc, sd] = (run(P, False), run(P, True))
S.apply(ident); rng = np.random.default_rng(0)
print("new, six-base mean: shipped stages / ORACLE run-split stages, each against shipped stages in recording order")
for sc in ["rec"] + S.SCHEMES:
    keys = [k for k in R if k[0] == sc]
    a = np.mean([[R[k][0][bn] - REF[bn] for bn in S.BASES] for k in keys], axis=(0, 1)); b = np.mean([[R[k][1][bn] - REF[bn] for bn in S.BASES] for k in keys], axis=(0, 1))
    rec = (b.mean() - a.mean()) / max(-a.mean(), 1e-9) if sc != "rec" else float("nan")
    print(f"  {sc:7s} shipped {a.mean():+.2f} | oracle run-split {b.mean():+.2f} (without the best {np.delete(b, int(np.argmax(b))).mean():+.2f}) | share of the loss recovered {rec:+.0%}", flush=True)
