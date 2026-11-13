"""SHUFPKG (ledger s169): v5.1b as uploaded under non-chronological delivery, returning people included: routing v4.1
(route4 max t10) + the own-space stage, gate calibration fixed from v5.1b in recording order. Every per-window store is
permuted (shuffle.py), each test session of the gate (rlib3.gate_model) is delivered in the same order, and the
session-keyed caches (bio streams, personal heads, own-space member streams) are cleared per order.
Members stay as computed in recording order (s165: re-streaming the SessionBN slots adds no loss for new people).
Usage: [SCHEMES=full,runs,within] [PERMS=2] [BASES=...] python studies/13nov/shufpkg.py"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("SCHEMES", "full,runs,within"); os.environ.setdefault("PERMS", "2")
import shuffle as S  # noqa: E402  (loads env: the same A as v5_replay's route4)
A, SC = S.A, S.SC; PR = S.PR
iB = [i for i, d in enumerate(S.DICTS) if d is A.pl.B][0]; IDX = {p: S.ORIG[iB][f"idx|{p}"].astype(int) for p in PR}
RANK = {"cur": None}; _gm0 = A.R3.gate_model


def gate_model_ordered(proto, f, oof, *a, **k):  # deliver each test session in the current order (global index -> rank)
    out = _gm0(proto, f, oof, *a, **k)
    if RANK["cur"] is not None:
        rk = RANK["cur"][proto]
        for s, (Xs, ys, gi) in list(out[0].items()):
            o = np.argsort(rk[np.asarray(gi).astype(int)], kind="stable"); out[0][s] = (Xs[o], np.asarray(ys)[o], np.asarray(gi)[o])
    return out


A.R3.gate_model = gate_model_ordered  # before v5_replay wraps it, so its NM records the delivery order
REPO = S.REPO; F_ = REPO + "/studies/12nov/v5_replay.py"; exec(compile(open(F_).read().split("\nres = []")[0], F_, "exec"))  # noqa: S102
BASES = S.BASES


def system(bn, P, oof):
    for c in (ns["_BIO"], A._H, STREAMS, REC):  # session-keyed caches depend on the delivery order
        c.clear()
    S.apply(P); RANK["cur"] = {p: _rank(IDX[p][P[p]]) for p in PR}
    mixes, sh, a = {}, {}, 0.0
    for p in PR:
        mixes[p], a = A.mix(bn, p, A.FAMS); _, sh[p] = A.shadow(mixes[p], p)
    out0 = ns["route4"](mixes, sh, a, sh["all_loso"] if oof is None else oof, {p: A.members(bn, p)["c1"] for p in PR}, "max", 10)[0]
    out = staged(out0)[0]; return {p: SC.person(out[p], p) for p in PR}, sh["all_loso"], len(REC)


def _rank(order):
    rk = np.full(int(order.max()) + 1, -1); rk[order] = np.arange(len(order)); return rk


if __name__ == "__main__":  # importable by toracle.py (system, S, PR, BASES)
    ident = {p: np.arange(S.N[p]) for p in PR}; REF, OOF, R = {}, {}, {}
    for bn in BASES:
        REF[bn], OOF[bn], nr = system(bn, ident, None)  # v5.1b in recording order; its new-people shadow fixes the gate's calibration
        print(f"{bn} recording order: new {REF[bn]['all_loso'].mean():.2f} seen {REF[bn]['seen_session'].mean():.2f} (P04 {REF[bn]['seen_session'][0]:.1f}, P10 {REF[bn]['seen_session'][1]:.1f}), routed sessions {nr}", flush=True)
    for sc in S.SCHEMES:
        for sd in range(S.PERMS):
            P = S.perm_for(sc, sd)
            for bn in BASES:
                R[sc, sd, bn], _, nr = system(bn, P, OOF[bn])
            print(f"  {sc} {sd}: routed sessions {nr} (last base)", flush=True)
    S.apply(ident); RANK["cur"] = None
    print("v5.1b against itself in recording order, six-base mean (new | seen: P04, P10 | J)")
    for sc in S.SCHEMES:
        dn = np.mean([[R[sc, sd, bn]["all_loso"] - REF[bn]["all_loso"] for bn in BASES] for sd in range(S.PERMS)], axis=(0, 1))
        ds = np.mean([[R[sc, sd, bn]["seen_session"] - REF[bn]["seen_session"] for bn in BASES] for sd in range(S.PERMS)], axis=(0, 1))
        print(f"  {sc:7s} new {dn.mean():+.2f} | seen {ds.mean():+.2f} (P04 {ds[0]:+.2f}, P10 {ds[1]:+.2f}) | J {0.5 * dn.mean() + 0.5 * ds.mean():+.2f}", flush=True)
    np.savez(REPO + "/runs/" + os.environ.get("SHUFPKG_OUT", "shufpkg.npz"), **{f"rec|0|{bn}|{p}": v[p] for bn, v in REF.items() for p in PR}, **{f"{sc}|{sd}|{bn}|{p}": v[p] for (sc, sd, bn), v in R.items() for p in PR})
