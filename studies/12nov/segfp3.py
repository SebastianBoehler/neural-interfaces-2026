"""SEGFP3 (ledger s62c): the routed systems (v3, v4.1) with every stage and gate run on cut or re-ordered streams.
A variant re-arranges the dev rows of each protocol (order within a session) and may split a session into streams.
Usage: python studies/12nov/segfp3.py"""
import os, sys
import numpy as np
REPO = os.path.abspath(__file__ + "/../../..")
R4 = REPO + "/studies/11nov/rt4/route4.py"; ns = {"__file__": R4, "__name__": "route4"}
sys.argv = ["route4.py", REPO + "/runs/replay_data/bio_model_psd_d64.npz"]
exec(compile(open(R4).read().split("\nbases = sys.argv")[0], R4, "exec"), ns)
A, R3, SC, pl = ns["A"], ns["R3"], ns["SC"], ns["pl"]; L = A.L; Z = L.Z
W = np.load(REPO + "/data/proteus_windows.npz", allow_pickle=True); ITF = W["interface"].astype(str)
POS = {}; STORES = [Z, SC.B, pl.B]; ORIG = [{k: v.copy() if isinstance(v, np.ndarray) else v for k, v in d.items()} for d in STORES]; _gm = R3.gate_model; KIND = ["session"]


def arrange(kind):
    """Rewrite every per-row array of both protocols: rows of a session in stream order, stream ids in sess|proto."""
    for d, o in zip(STORES, ORIG):
        d.clear(); d.update({k: v.copy() if isinstance(v, np.ndarray) else v for k, v in o.items()})
    for p in A.PROTOS:
        idx = ORIG[0][f"idx|{p}"].astype(int); s0 = ORIG[0][f"sess|{p}"]; n = len(idx); itf = ITF[idx]
        order = np.lexsort((np.arange(n), _key(kind, idx, s0), s0))
        for d in STORES:
            for k, v in list(d.items()):
                if not isinstance(v, np.ndarray) or not k.endswith("|" + p):
                    continue
                if v.ndim >= 1 and v.shape[0] == n:
                    d[k] = v[order]
                elif v.ndim >= 2 and v.shape[1] == n:
                    d[k] = v[:, order]
        if kind == "reset itf":
            new = np.unique(np.char.add(np.char.add(s0[order].astype(str), "|"), itf[order]), return_inverse=True)[1]
            for d in STORES:
                if f"sess|{p}" in d:
                    d[f"sess|{p}"] = new
    A.C1.update({p: pl.get("B2_pool", "core1", p) for p in A.PROTOS}); A._H.clear(); A._HEADS.clear(); ns["_BIO"].clear(); KIND[0] = kind
    POS.update({p: {int(g): k for k, g in enumerate(Z[f"idx|{p}"].astype(int))} for p in A.PROTOS})


RUNK = np.char.add(W["task"].astype(str), W["run"].astype(str)); ONSET = W["onset"].astype(float)


def _key(kind, idx, s0):
    """Sort key within a session (ties broken by acquisition order)."""
    n = len(idx); itf = ITF[idx]; rng = np.random.default_rng(0)
    if kind in ("reset itf", "order Graz,BH"):
        return (itf != "Graz").astype(int)
    if kind == "order BH,Graz":
        return (itf == "Graz").astype(int)
    if kind == "O1 session shuffle":
        return rng.random(n)
    if kind == "O6 by onset":
        return ONSET[idx]
    if kind in ("O2 shuffle in run", "O3 runs permuted"):
        key = np.zeros(n); rk = np.char.add(np.char.add(s0.astype(str), "|"), RUNK[idx])
        for j, u in enumerate(dict.fromkeys(rk)):
            m = np.flatnonzero(rk == u)
            key[m] = (j + rng.random(len(m)) * 0.9) if kind == "O2 shuffle in run" else rng.random()
        return key
    return np.zeros(n)


def gate_model(proto, f, oof):
    te, names, Gn, Q, per, x = _gm(proto, f, oof); out = {}
    for s, (X, y, gi) in te.items():
        gi = np.asarray(gi); itf = ITF[gi]; k = KIND[0]
        if k == "session":
            out[s] = (X, y, gi); continue
        if k.startswith("O"):  # the stream order of this session, taken from the re-arranged protocol rows
            o = np.argsort([POS[proto][int(g)] for g in gi]); out[s] = (X[o], y[o], gi[o]); continue
        first = "BrainHero" if k == "order BH,Graz" else "Graz"; o = np.r_[np.flatnonzero(itf == first), np.flatnonzero(itf != first)]
        if k == "reset itf":
            for name in (first, "other"):
                m = np.flatnonzero(itf == first) if name == first else np.flatnonzero(itf != first); out[f"{s}#{name}"] = (X[m], y[m], gi[m])
        else:
            out[s] = (X[o], y[o], gi[o])
    return out, names, Gn, Q, per, x


R3.gate_model = gate_model
Jf = lambda o: 0.5 * SC.person(o["all_loso"], "all_loso").mean() + 0.5 * SC.person(o["seen_session"], "seen_session").mean()  # noqa: E731
for bn in ("B2_pool", "B2_s345"):
    ref = {}
    for kind in (("session", "O1 session shuffle", "O2 shuffle in run", "O3 runs permuted", "O6 by onset") if os.environ.get("SEGFP_ORDERS") else ("session", "reset itf", "order Graz,BH", "order BH,Graz")):
        arrange(kind); A.C1.update({p: pl.get("B2_pool", "core1", p) for p in A.PROTOS})
        v3, i3 = ns["system4"](bn, "v3", 10); v4, i4 = ns["system4"](bn, "max", 10)
        sh = {p: A.shadow(A.mix(bn, p, A.FAMS)[0], p)[1] for p in A.PROTOS}  # unrouted GEB tail: the TSEM-package level
        pe, pt = {}, {}
        for p in A.PROTOS:
            _, _, pe[p], pt[p] = L.chain(bn, p, stages=True)
        r = dict(tsem=Jf(sh), em0=Jf(pe), ts0=Jf(pt), v3=Jf(v3), v4=Jf(v4), n3=(len([i for i in i3 if i[0] == "all_loso"]), len([i for i in i3 if i[0] == "seen_session"])),
                 n4=(len([i for i in i4 if i[0] == "all_loso"]), len([i for i in i4 if i[0] == "seen_session"])))
        if kind == "session":
            ref = r; print(f"{bn} per session, acquisition order: unrouted {r['tsem']:.2f}, v3 {r['v3']:.4f}, v4.1 {r['v4']:.2f} (v4.1 - v3 {r['v4'] - r['v3']:+.2f}); streams routed v3 new/seen {r['n3']}, v4.1 {r['n4']}", flush=True)
            continue
        ok = abs((r["v3"] - ref["v3"]) + 1.05) <= 0.4 and abs((r["v4"] - r["v3"]) - 0.27) <= 0.25 and abs((r["em0"] - ref["em0"]) + 0.72) <= 0.4 and abs((r["ts0"] - ref["ts0"]) + 1.17) <= 0.4
        print(f"   {kind:18s}: plain SessionEM level {r['em0'] - ref['em0']:+.2f} (sealed -0.72) | plain TSEM level {r['ts0'] - ref['ts0']:+.2f} (sealed -1.17)", end=" || ")
        print(f"   {kind:14s}: unrouted {r['tsem'] - ref['tsem']:+.2f} | v3 {r['v3'] - ref['v3']:+.2f} (sealed -1.05) | v4.1 - v3 {r['v4'] - r['v3']:+.2f} (sealed +0.27) {'SUPPORTED' if ok else '         '} | "
              f"v4.1 {r['v4'] - ref['v4']:+.2f} | streams routed v3 {r['n3']}, v4.1 {r['n4']}", flush=True)
arrange("session")
