"""CHAINBAG (ledger s61): averaging the final posteriors of several complete chains, and of several stage settings.
Usage: python studies/12nov/chainbag.py"""
import os, sys
import numpy as np
REPO = os.path.abspath(__file__ + "/../../..")
sys.path[:0] = [REPO + "/replay"]
import env  # noqa: E402
A = env.A; SC = A.SC; L = A.L
J = lambda o: 0.5 * SC.person(o["all_loso"], "all_loso").mean() + 0.5 * SC.person(o["seen_session"], "seen_session").mean()  # noqa: E731
amean = lambda outs: {p: np.mean([o[p] for o in outs], 0) for p in A.PROTOS}  # noqa: E731


def gmean(outs):
    r = {}
    for p in A.PROTOS:
        Lg = np.mean([np.log(np.clip(o[p], 1e-9, 1)) for o in outs], 0); e = np.exp(Lg - Lg.max(1, keepdims=True)); r[p] = e / e.sum(1, keepdims=True)
    return r


C = {bn: A.system(bn, A.FAMS)["out"] for bn in env.BASES}; print("single chains: " + " ".join(f"{bn} {J(o):.2f}" for bn, o in C.items()), flush=True)
for name, parts, ref in (("(a) s0 + s1 + s2", ("B2_s0", "B2_s1", "B2_s2"), "B2_s012"), ("(b) s012 + s345", ("B2_s012", "B2_s345"), "B2_pool")):
    for tag, f in (("arithmetic", amean), ("geometric", gmean)):
        o = f([C[b] for b in parts]); r = SC.compare(o, C[ref])
        print(f"{name} {tag}: J {J(o):.2f} | vs the pooled chain {ref} ({J(C[ref]):.2f}): dJ {r['J']:+.2f} new {r['new']:+.2f} seen {r['seen']:+.2f} new>=0 {r['npos']}/10 | "
              f"vs the mean single chain ({np.mean([J(C[b]) for b in parts]):.2f}): {J(o) - np.mean([J(C[b]) for b in parts]):+.2f}", flush=True)
lik0, reb0 = L.G.tsem_lik, L.rebal; RHO, WIN = [0.7], [30]
L.G.tsem_lik = lambda P, sess, X, rho=0.7, **kw: lik0(P, sess, X, rho=RHO[0], **kw)
L.rebal = lambda P, sess, alpha=1.15, k=5.0, W=30, **kw: reb0(P, sess, alpha=alpha, k=k, W=WIN[0], **kw)
if "fresh" in sys.argv:  # s61b: the rebalancing-window bag on the fresh bases, new people
    for bn in env.FRESH:
        outs = []
        for w in (25, 30, 35):
            WIN[0] = w; outs.append(A.system(bn, A.FAMS)["out"])
        r = SC.compare(amean(outs), outs[1]); print(f"FRESH {bn}: new {r['new']:+.2f} seen {r['seen']:+.2f} new>=0 {r['npos']}/10 | " + " ".join(f"{q[1:]}:{v:+.1f}" for q, v in zip(SC.PEOPLE, r["dn"])), flush=True)
    WIN[0] = 30; L.G.tsem_lik, L.rebal = lik0, reb0; sys.exit(0)
for bn in env.EXPL:
    outs = {}
    for rho in (0.6, 0.7, 0.8):
        for w in (25, 30, 35):
            RHO[0], WIN[0] = rho, w; outs[(rho, w)] = A.system(bn, A.FAMS)["out"]
    base = outs[(0.7, 30)]; assert abs(J(base) - J(C[bn])) < 1e-9, "the patched settings do not reproduce the shipped chain"
    print(f"(c) {bn}: single settings J " + " ".join(f"{k[0]}/{k[1]}:{J(o):.2f}" for k, o in outs.items()), flush=True)
    for tag, keys in (("all nine", list(outs)), ("rho only", [(r, 30) for r in (0.6, 0.7, 0.8)]), ("W only", [(0.7, w) for w in (25, 30, 35)])):
        for nm, f in (("arithmetic", amean), ("geometric", gmean)):
            r = SC.compare(f([outs[k] for k in keys]), base); print(f"    bag of {tag} ({nm}): dJ {r['J']:+.2f} new {r['new']:+.2f} seen {r['seen']:+.2f} new>=0 {r['npos']}/10", flush=True)
RHO[0], WIN[0] = 0.7, 30; L.G.tsem_lik, L.rebal = lik0, reb0
