"""s247a: the routed chain (pview chain.py, s214 form, arms ER and T) with session 2 of the tested person made public.
For the seen_session fold of q (base nets / core1 trained without q's sessions 2 and 3), everything the routed path builds
from q's public data gets session 2 added: the v2 gate's gallery and basis (rlib3.gate_model's per-person sessions), the bio
gate's enrolment rows, the personal head (ps1_heads on per), T's prior (pri from per) and the own-space stage
(ownspace.Personal on sessions 1 and 2). Session 2's core1 PCs are the fold's stored causal test features. Population
members, core1's basis, the eye and rest nets stay as in seen_session. Only session 3 is meaningful (session 2 is leaky).
S2=0: the same code path with nothing added; must reproduce runs/chainout/s245_ret.npz (asserted per order and base).
Also saves the gate's ramp r per window (for fusion.py).
Usage: S2=1|0 [ORDERS=..] [BASES=..] python studies/14nov/strongpersonal/chain12.py -> runs/strongpersonal/chain12_s2<S2>.npz"""
import os, sys, time
import numpy as np
os.environ.setdefault("ARMS", "ER,T")
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE + "/../pview")
import chain as CH  # noqa: E402  (pview chain.py: exec'd retv6 head, the T prior hook, system())
S2 = os.environ.get("S2", "1") == "1"; R6, K, A, ns, p, Z, D = CH.R6, CH.K, CH.A, CH.ns, CH.p, CH.Z, CH.D
R3 = A.R3; L, R2, RS = R3.L, R3.R2, R3.RS
ORDERS = os.environ.get("ORDERS", ",".join([f"within{k}" for k in range(8)] + [f"full{k}" for k in range(4)] + [f"runs{k}" for k in range(4)])).split(",")
BASES = os.environ.get("BASES", "B2_pool,B2_s345,B2_s012,B2_s0,B2_s1,B2_s2").split(",")
_gm_orig = K._gm0; ADDED = {}


def own(f):  # fold "P04_s23" -> "P04"
    return f[:3] if p == "seen_session" and f.endswith("_s23") else None


def gate_model_s12(proto, f, oof, npc=256):
    """rlib3.gate_model with q's session 2 (the fold's stored test features) as one more public session of q."""
    if proto != "seen_session" or not S2:
        return _gm_orig(proto, f, oof, npc)
    tr, te = L.fold(proto, f); itr = np.load(f"{L.FE}/{proto}-{f}.npz")["idx_tr"]; st = L.STREAM[itr]; per, bias = {}, {}
    q = own(f); s2 = [s for s in te if s.endswith(f"/{q}|2")]; assert len(s2) == 1, (f, list(te))
    X2, y2, g2 = te[s2[0]]; items = [(s, X, y, itr[st == s]) for s, (X, y) in tr.items()] + [(s2[0], X2, y2, np.asarray(g2))]
    for s, X, y, rows in items:
        per.setdefault(L.PER(s), []).append((X, y)); Po = oof[rows]
        bias.setdefault(L.PER(s), []).append(R2.contrast(X, L.weights(Po, len(y), "q70", y), npc) - R2.contrast(X, R2.onehot(y), npc))
    names = list(per); assert L.PER(s2[0]) in names and len(per[L.PER(s2[0])]) == 2, "q must have sessions 1 and 2"
    gal = np.stack([np.mean([R2.contrast(X, R2.onehot(y), npc) for X, y in per[n]], 0) for n in names])
    pop = gal.mean(0); b = np.mean([np.mean(bias[n], 0) for n in names], 0); Q = R2.basis(pop, b); ADDED["gate", f] = len(y2)
    return te, names, RS.gal_norm(gal, Q), Q, per, gal


def bio_stream_s12(proto, f, s, gi, names, t0):  # = route4.bio_stream with q's session 2 among the enrolment rows
    key = (proto, f, s, t0); _BIO = ns["_BIO"]
    if key not in _BIO:
        itr = np.load(f"{ns['RL'].FE}/{proto}-{f}.npz")["idx_tr"]
        if S2 and own(f):
            add = np.flatnonzero((ns["PER"] == own(f)) & (np.char.endswith(ns["STR"], "|2"))); assert len(add) > 150; itr = np.r_[itr, add]; ADDED["bio", f] = len(add)
        b = ns["Bio"](ns["fold_params"](ns["MODEL"], ns["FB"][itr], ns["STR"][itr], list(names), t0=t0, tau=ns["TAU"]))
        r, j, sc = np.zeros(len(gi)), np.full(len(gi), -1), np.full(len(gi), np.nan)
        for t, g in enumerate(gi):
            j[t], r[t] = b.step(ns["X"][g]); sc[t] = b.last[1] if b.last else np.nan
        _BIO[key] = (r, j, sc)
    return _BIO[key]


MOD = {}


def members_s12(proto, s, q):  # = v5_replay.members with q's session 2 among the own-space training sessions
    names, gi, f = K.NM[(proto, s)]
    if (proto, s, q) not in K.STREAMS:
        if (proto, f, q) not in MOD:
            O = K.O; itr = np.load(f"{K.RL.FE}/{proto}-{f}.npz")["idx_tr"]; itr = itr[O.PER[itr] == q[-3:]]
            if S2 and own(f) == q[-3:]:
                itr = np.r_[itr, np.flatnonzero((O.PER == q[-3:]) & (O.SES == "2"))]; ADDED["own", f] = len(itr)
            MOD[(proto, f, q)] = O.Personal([itr[O.SES[itr] == u] for u in sorted(set(O.SES[itr]))])
        K.STREAMS[(proto, s, q)] = MOD[(proto, f, q)].stream(gi)
    return K.STREAMS[(proto, s, q)]


K._gm0, ns["bio_stream"], K.members = gate_model_s12, bio_stream_s12, members_s12


def main():
    t0 = time.time(); OOF = {bn: R6["oof_rec"](bn) for bn in BASES}; OUT = {}; ref = np.load(f"{D.REPO}/runs/chainout/s245_ret.npz")
    for order in ORDERS:
        P = R6["perm"](order); Pp = P[p]; inp = R6["inputs"](order)
        for c in (ns["_BIO"], A._H, K.STREAMS, K.REC):
            c.clear()
        A._HEADS.clear(); CH.S.apply(P); K.RANK["cur"] = {q: K._rank(K.IDX[q][P[q]]) for q in K.PR}
        R6["apply_members"](R6["member_deltas"](order), Pp); POC = {"orig": CH.personal_streams(Pp)}; back = np.argsort(Pp)
        for bn in BASES:
            outs, rrs = {}, {}
            for arm in ("ER", "T"):
                outs[arm], rrs[arm] = CH.system(bn, arm, Pp, OOF[bn], inp, POC)
            ramp = np.zeros(len(Pp))
            for (rows, r, q), (rows_e, r_e, _) in zip(rrs["T"].values(), rrs["ER"].values()):
                assert np.array_equal(rows, rows_e) and np.array_equal(r, r_e), "T: routing differs from ER's"
                outs["T"][rows[:int(np.flatnonzero(r > 0)[0])]] = outs["ER"][rows[:int(np.flatnonzero(r > 0)[0])]]; ramp[rows] = r
            for arm in ("ER", "T"):
                OUT[f"{order}|{bn}|{arm}"] = np.asarray(outs[arm])[back].astype(np.float32)
                if not S2:
                    d_ = np.abs(OUT[f"{order}|{bn}|{arm}"] - ref[f"{order}|{bn}|{arm}"]).max(); assert d_ < 1e-5, f"S2=0 {order} {bn} {arm}: differs from s245 by {d_}"
            OUT[f"{order}|{bn}|ramp"] = ramp[back].astype(np.float32)
            print(f"  {order} {bn} done [{time.time() - t0:.0f}s]{' (= s245)' if not S2 else ''} added {dict(ADDED)}", flush=True)
    os.makedirs(f"{D.REPO}/runs/strongpersonal", exist_ok=True)
    f = f"{D.REPO}/runs/strongpersonal/chain12_s2{int(S2)}" + ("" if len(ORDERS) == 16 and len(BASES) == 6 else "_smoke") + ".npz"
    np.savez_compressed(f, **OUT); print("written", f, flush=True)


if __name__ == "__main__":
    main()
