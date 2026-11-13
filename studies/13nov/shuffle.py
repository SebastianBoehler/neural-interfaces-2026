"""SHUF (ledger s164): the chain under non-chronological delivery within a session (organisers, 13 Nov: the sealed
stream is not recording order). Every per-window array of the replay (members, views, labels, cells, sessions) is
permuted within each session and the stages are rerun; scored per person, which does not depend on order.
Schemes (the sealed scheme is unknown):
  rec     recording order (identity; must reproduce the chain exactly)
  full    a uniform random permutation of the session's windows
  runs    runs in a random order, recording order inside each run
  within  runs in recording order, windows shuffled inside each run
Stage by stage for new people (all_loso): member mix -> rebalancing -> gated SessionEM -> gated TSEM (gm.pipe_g, the
unrouted tail every new-person window takes). LIMIT: members are fixed as computed in recording order (their own
causal session statistics, e.g. SessionBN, core1's running recentring, the eye net's standardisation, are not
re-streamed). This measures the stages' order dependence, not the members'.
Usage: [SCHEMES=full,runs,within] [PERMS=5] [BASES=...] python studies/13nov/shuffle.py"""
import os, sys
import numpy as np
REPO = os.path.abspath(__file__ + "/../../..")
sys.path.insert(0, REPO + "/replay")
import env  # noqa: E402
A = env.A; SC = A.SC; gm = A.gm; PR = A.PROTOS; p0 = "all_loso"
N = {p: len(A.L.Z[f"y|{p}"]) for p in PR}
SCHEMES = os.environ.get("SCHEMES", "full,runs,within").split(","); PERMS = int(os.environ.get("PERMS", 5))
BASES = os.environ.get("BASES", ",".join(env.BASES)).split(",")
DICTS = {id(d): d for m in list(sys.modules.values()) for d in vars(m).values() if isinstance(d, dict) and f"y|{p0}" in d and isinstance(d.get(f"y|{p0}"), np.ndarray)}.values()
DICTS = list(DICTS); assert any(d is A.L.Z for d in DICTS) and any(d is A.pl.B for d in DICTS), "the replay stores were not found"
ORIG = [{k: v for k, v in d.items()} for d in DICTS]; C1 = {p: A.C1[p].copy() for p in PR}
with np.load(REPO + "/data/proteus_windows.npz") as w:
    RUN = np.char.add(w["task"].astype(str), np.char.add("|", w["run"].astype(str)))
print(f"SHUF: {len(DICTS)} stores permuted; schemes {SCHEMES} x {PERMS} permutations; members fixed as computed in recording order", flush=True)


def perm_for(scheme, seed):
    out = {}
    for p in PR:
        sess = ORIG[0][f"sess|{p}"] if f"sess|{p}" in ORIG[0] else A.L.Z[f"sess|{p}"]; run = RUN[A.pl.B[f"idx|{p}"].astype(int)] if False else RUN[ORIG[[i for i, d in enumerate(DICTS) if d is A.pl.B][0]][f"idx|{p}"].astype(int)]
        r = np.random.default_rng([seed, PR.index(p), SCHEMES.index(scheme) if scheme in SCHEMES else 99]); P = np.arange(N[p])
        for s in dict.fromkeys(sess):
            rows = np.flatnonzero(sess == s); assert (np.diff(rows) == 1).all(), "a session's rows are not contiguous"
            if scheme == "full":
                P[rows] = rows[r.permutation(len(rows))]
            elif scheme in ("runs", "within"):
                blocks = [rows[run[rows] == u] for u in dict.fromkeys(run[rows])]
                if scheme == "runs":
                    blocks = [blocks[i] for i in r.permutation(len(blocks))]
                else:
                    blocks = [b[r.permutation(len(b))] for b in blocks]
                P[rows] = np.concatenate(blocks)
        out[p] = P
    return out


def apply(P):
    for d, o in zip(DICTS, ORIG):
        for k, v in o.items():
            p = k.split("|")[-1]
            if p in N and isinstance(v, np.ndarray):
                ax = [i for i, s in enumerate(v.shape) if s == N[p]]
                d[k] = np.take(v, P[p], axis=ax[0]) if ax else v
    for p in PR:
        A.C1[p] = C1[p][P[p]]


def stages(bn):
    mix = A.mix(bn, p0, A.FAMS)[0]; b = gm.M.rebal(mix, p0, None); g = gm.gate(b, p0); e = gm.em_g(b, p0, g); t = gm.tsem_g(e, p0, g)
    return {k: SC.person(v, p0) for k, v in (("mix", mix), ("rebal", b), ("sessionem", e), ("tsem", t))}


if __name__ != "__main__":  # imported as a library (retune.py): stores, perm_for, apply, stages; no run
    pass
elif os.environ.get("DUMP_PERMS"):  # s165: the delivery orders as global window indices, for re-streaming members on GPU
    IDX0 = ORIG[[i for i, d in enumerate(DICTS) if d is A.pl.B][0]]
    np.savez(os.environ["DUMP_PERMS"], **{f"{sc}|{sd}|{p}": IDX0[f"idx|{p}"].astype(int)[perm_for(sc, sd)[p]] for sc in SCHEMES for sd in range(PERMS) for p in PR})
    sys.exit(print("dumped", os.environ["DUMP_PERMS"]))
else:
    ident = {p: np.arange(N[p]) for p in PR}
    REF = {}
    for bn in BASES:
        apply(ident); REF[bn] = stages(bn)
        assert np.allclose(SC.person(gm.pipe_g(A.mix(bn, p0, A.FAMS)[0], p0), p0), REF[bn]["tsem"]), "stages() does not equal gm.pipe_g"
    print("recording order, new people: " + " | ".join(f"{bn} " + " ".join(f"{k} {v.mean():.2f}" for k, v in REF[bn].items()) for bn in BASES[:2]), flush=True)
    R = {}
    for sc in SCHEMES:
        for seed in range(PERMS):
            P = perm_for(sc, seed)
            for bn in BASES:
                apply(P); R[sc, seed, bn] = stages(bn)
                assert np.allclose(R[sc, seed, bn]["mix"], REF[bn]["mix"]), f"{sc}: the per-window mix scored differently, the permutation bookkeeping is broken"
        d = {k: np.mean([[R[sc, s, bn][k].mean() - REF[bn][k].mean() for bn in BASES] for s in range(PERMS)]) for k in REF[BASES[0]]}
        rng_ = [np.mean([R[sc, s, bn]["tsem"].mean() - REF[bn]["tsem"].mean() for bn in BASES]) for s in range(PERMS)]
        print(f"{sc:7s}: against recording order, new, mean over bases and permutations: " + " ".join(f"{k} {v:+.2f}" for k, v in d.items())
              + f" | chain per permutation {min(rng_):+.2f} to {max(rng_):+.2f}", flush=True)
    apply(ident)
    np.savez(REPO + "/runs/shuffle_13nov.npz", **{f"{sc}|{s}|{bn}|{k}": v for (sc, s, bn), st in R.items() for k, v in st.items()}, **{f"rec|0|{bn}|{k}": v for bn, st in REF.items() for k, v in st.items()})
