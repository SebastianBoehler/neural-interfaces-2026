"""RETV6 (ledger s186 item 4): the three v6 additions on top of the ROUTED path v5.1b gives returning people
(seen_session: P04, P10), against routed v5.1b (studies/13nov/shufpkg.py) under the same delivery order.
Chain as shufpkg: route4 "max" t10 (shadow -> v3 gate | bio gate -> personal head slotted into the mix -> gm.pipe_g on
routed sessions) + the own-space stage; gate calibration (oof) fixed from v5.1b in recording order; members as recorded.
Arms (each a change of the seen_session mix / view, so it reaches the shadow, the v3 gate's input and the routed tail):
  base  routed v5.1b (asserted equal to runs/shufpkg.npz)
  E     eye net: 3 log-posterior columns appended to SessionEM's GE view (shadow and routed tail)
  R     rest member at weight 1 in the mix (15 -> 16; the personal head's share a = core1's share follows: 1/16)
  F     gC5 delta1: each SessionBN slot (5-8) + (gC5 - base) of the base's first seed, on every row (unmasked; pkg.py's
        ship form masks FT off on routed sessions, which here is every dev session)
  ERF   all three
Inputs in the delivery order they were streamed in: rec = ttt7 / ocunet s0-2 / rest4 s0-2; withinK = ttt18withinK /
ocushuf/withinK (all seeds) / restshuf/withinK (all seeds). Only seen_session is run (RR.FOLDS narrowed per call).
Usage: [BASES=B2_s0] [ORDERS=rec,within0,within1] python studies/13nov/retv6.py"""
import glob, hashlib, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import shufpkg as K  # noqa: E402  (S, A, ns, staged, REC, NM, STREAMS, RANK, _rank, IDX)
S, A, SC, ns = K.S, K.A, K.SC, K.ns; Z = A.L.Z; PR = K.PR; p = os.environ.get("RT_PROTO", "seen_session"); REPO = S.REPO; IDX = K.IDX[p]
# s201: RT_PROTO=all_loso runs the same arms on new people (to see whether the additions change who gets routed);
# GATE_V51B=1 computes the shadow (the v3 gate's input, and the output before routing) from v5.1b's unchanged mix and view,
# so the additions reach only the routed tail (the gate never sees them, as in v6f/v6.1)
GATE_V51B = bool(os.environ.get("GATE_V51B"))
SLOT = {"fbraw": 5, "fb": 6, "sinc": 7, "fbl": 8}; K0 = {"B2_s0": 0, "B2_s1": 1, "B2_s2": 2, "B2_s012": 0, "B2_s345": 3, "B2_pool": 0}  # delta1: first base seed
BASES = os.environ.get("BASES", "B2_pool,B2_s345,B2_s012,B2_s0,B2_s1,B2_s2").split(","); ORDERS = os.environ.get("ORDERS", "rec,within0,within1").split(",")
ARMS = ["base", "E", "R", "ER", "F", "ERF"]; REF = np.load(REPO + "/runs/shufpkg.npz")  # ER (s199): eye view + rest member, no FT
# s190: LABEL_RT=1 writes runs/labels/chain_rt_<order>.npz (new people: chain_v51b_<order>; returning: routed v5.1b as emitted)
# and exits; FT_SFX=T reads ttt18<order>T (gC5 fine-tuned on those labels) for arm F instead of ttt18<order>
LABEL_RT, FT_SFX, LABRT = bool(os.environ.get("LABEL_RT")), os.environ.get("FT_SFX", ""), {}
OCU_SS = os.environ.get("OCU_SS", "runs/ocushuf_ss")  # s214: eye outputs re-streamed in the protocol's own order (order_sha)
FT_FORM = os.environ.get("FT_FORM", "gC5")  # s198: a form of the stack run (FT_SFX=KF), e.g. s03, c01, e02
if LABEL_RT:
    BASES, ARMS = ["B2_pool"], ["base"]
PERMS_SHA = "6046bf5b2231d29d28bbfcf88ea7665d4aaa506c5787bd836ebec6a5b945430f"
assert hashlib.sha256(open(f"{REPO}/runs/shuffle_perms.npz", "rb").read()).hexdigest() == PERMS_SHA, "runs/shuffle_perms.npz is not the pinned order file"
PF = np.load(f"{REPO}/runs/shuffle_perms.npz")


def ld(path):
    z = np.load(path); assert (z["index"].astype(int) == IDX).all(), path; return z["proba"]


def mean_of(files, what):
    assert files, f"no {what} files"; return np.mean([ld(f) for f in files], 0), len(files)


def inputs(order):
    """-> FT deltas per (arch, seed), eye-net log view, rest member; all in recording index layout."""
    if order == "rec":
        ft = {(a, s): ld(f"{REPO}/runs/ttt/probs7/ttt7_{a}_gC5-{p}-s{s}.npz") - ld(f"{REPO}/runs/ttt/probs7/ttt7_{a}_base-{p}-s{s}.npz") for a in SLOT for s in range(6)}
        oc, no = mean_of([f"{REPO}/runs/ocunet/probs/oc_nomix-{p}-s{s}.npz" for s in range(3)], "eye")
        rs, nr = mean_of([f"{REPO}/runs/rest4/probs/rest4_sh_r1-{p}-s{s}.npz" for s in range(3)], "rest")
    else:
        d = f"{REPO}/runs/ttt/probs18/ttt18{order}{FT_SFX}"
        ft = {(a, s): ld(f"{d}_{a}_{FT_FORM}-{p}-s{s}.npz") - ld(f"{d}_{a}_base-{p}-s{s}.npz") for a in SLOT for s in range(6)}
        ofs = sorted(glob.glob(f"{REPO}/{OCU_SS}/{order}/oc_nomix-{p}-s*.npz")); want = hashlib.sha256(np.ascontiguousarray(PF[f"{order.rstrip('0123456789')}|{order[len(order.rstrip('0123456789')):]}|{p}"].astype(np.int64)).tobytes()).hexdigest()
        for f_ in ofs:  # s214 (P0-a): the eye net must have been standardised in THIS protocol's delivery order
            z_ = np.load(f_); assert "order_sha" in z_.files and str(z_["order_sha"]) == want, f"{f_}: eye net not in {p}'s delivery order of {order}"
        oc, no = mean_of(ofs, "eye")
        rs, nr = mean_of(sorted(glob.glob(f"{REPO}/runs/restshuf/{order}/rest4_sh_r1-{p}-s*.npz")), "rest")
    print(f"{order}: FT gC5 delta1 ({'ttt7' if order == 'rec' else 'ttt18' + order}), eye net {no} seeds, rest net {nr} seeds", flush=True)
    return ft, np.log(np.clip(oc, 1e-3, 1)), rs


RT_MEMBERS = os.environ.get("RT_MEMBERS", "1") == "1"  # s214 (bugs_scoring P0-b): returning people's members re-streamed too
BSD = {"B2_s0": [0], "B2_s1": [1], "B2_s2": [2], "B2_s012": [0, 1, 2], "B2_s345": [3, 4, 5], "B2_pool": [0, 1, 2, 3, 4, 5]}


def holder(pat, key, what):  # the single file among pat's matches holding key; none -> loud failure (no silent recording order)
    hit = [f for f in sorted(glob.glob(f"{REPO}/runs/{pat}")) if key in np.load(f)]; assert len(hit) == 1, f"{what}: {len(hit)} files hold {key}"
    return np.load(hit[0])


def member_deltas(order):
    """seen_session changes (delivery order - own recording-order rebuild) of every input shufcombo re-streams for new people."""
    sc, sd = order.rstrip("0123456789"), order[len(order.rstrip("0123456789")):]
    CM, HB, TS = holder("coremember*.npz", f"{order}|{p}", "core1"), holder("hilbertmember*.npz", f"{order}|{p}", "Hilbert"), holder("teachshuf*.npz", f"{order}|{p}", "teacher")
    csf = [f for f in sorted(glob.glob(f"{REPO}/runs/coreshuf_p{sd}.npz") + glob.glob(f"{REPO}/runs/coreshuf_p{sd}_*.npz")) if f"{sc}|{p}" in np.load(f)]
    assert len(csf) == 1, f"pc128: {len(csf)} files hold {sc}|{p} for seed {sd}"; CS = np.load(csf[0]); GS = np.load(f"{REPO}/runs/geshuf.npz"); ref = S.ORIG[[i for i, d in enumerate(S.DICTS) if d is Z][0]]
    sg = np.sign([np.corrcoef(ref[f"pc128|{p}"][:, j], CS[f"rec|{p}"][:, j])[0, 1] for j in range(ref[f"pc128|{p}"].shape[1])])
    assert np.allclose(ref[f"GE|{p}"][:, 32:], GS["rec"][IDX]), "GE columns 32: are not the event rates"
    d = f"{REPO}/runs/ttt/probs18/ttt18{order}{FT_SFX}"
    fr = {(a, k): ld(f"{d}_{a}_base-{p}-s{k}.npz") - ld(f"{REPO}/runs/ttt/probs7/ttt7_{a}_base-{p}-s{k}.npz") for a in SLOT for k in range(6)}
    return dict(c1=CM[f"{order}|{p}"] - CM[f"rec|{p}"], hb=HB[f"{order}|{p}"] - HB[f"rec|{p}"], ts=TS[f"{order}|{p}"] - TS[f"rec|{p}"],
                pc=sg * (CS[f"{sc}|{p}"] - CS[f"rec|{p}"]), ge=GS[f"{order}|{p}"][IDX] - GS["rec"][IDX], fr=fr)


def apply_members(D, Pp):  # after S.apply(P): every array of protocol p is in delivery order, fresh copies of the stored ones
    A.C1[p] = A.C1[p] + D["c1"][Pp]
    Z[f"pc128|{p}"] = Z[f"pc128|{p}"] + D["pc"][Pp].astype(Z[f"pc128|{p}"].dtype)
    Z[f"GE|{p}"] = Z[f"GE|{p}"].copy(); Z[f"GE|{p}"][:, 32:] = Z[f"GE|{p}"][:, 32:] + D["ge"][Pp]
    for bn in BASES:
        Ps = Z[f"{bn}|Ps|{p}"].copy(); Ps[0] = Ps[0] + (D["c1"][Pp] + D["hb"][Pp]) / 2; Ps[10] = Ps[10] + D["ts"][Pp]
        for a, sl in SLOT.items():
            x = np.clip(Ps[sl] + np.mean([D["fr"][a, k] for k in BSD[bn]], 0)[Pp], 1e-6, None); Ps[sl] = x / x.sum(1, keepdims=True)
        Z[f"{bn}|Ps|{p}"] = Ps


def perm(order):
    if order == "rec":
        return {q: np.arange(S.N[q]) for q in PR}
    sc, sd = order[:-1], int(order[-1]); P = S.perm_for(sc, sd)
    assert (IDX[P[p]] == PF[f"{sc}|{sd}|{p}"]).all(), "the stages' order is not the order the inputs were streamed in"
    return P


def oof_rec(bn):  # v5.1b's new-people shadow in recording order: the gate's fixed calibration (as shufpkg)
    S.apply({q: np.arange(S.N[q]) for q in PR}); K.RANK["cur"] = None
    return A.shadow(A.mix(bn, "all_loso", A.FAMS)[0], "all_loso")[1]


def system(bn, arm, Pp, oof, inp):
    ft, ocu, rest = inp; fams, extra, ge0, folds0 = list(A.FAMS), [], Z[f"GE|{p}"], A.RR.FOLDS
    if "F" in arm:
        for a_, sl in SLOT.items():
            base = A.members(bn, p)[sl]; x = np.clip(base + ft[a_, K0[bn]][Pp], 1e-6, None); x = x / x.sum(1, keepdims=True)
            assert np.abs(x - base).max() > 1e-3, f"FT {a_} is a no-op"
            A.EXTRA[f"rt_{a_}"] = {p: x}; fam = next(f for f, it in A.FAM.items() if any(k == sl for k, _ in it))
            if fam in fams:
                fams.remove(fam); extra += list(A.FAM[fam])
            extra = [(k, w) for k, w in extra if k != sl] + [(f"rt_{a_}", 1.0)]
    if "R" in arm:
        A.EXTRA["rt_rest"] = {p: rest[Pp]}; extra += [("rt_rest", 1.0)]
    if "E" in arm:
        Z[f"GE|{p}"] = np.c_[ge0, ocu[Pp]]
    try:
        A.RR.FOLDS = {p: folds0[p]}; K.REC.clear()  # stale REC entries would leak into staged(); only seen_session is run
        mix, a = A.mix(bn, p, fams, extra)
        if GATE_V51B:  # the gate and the pre-routing output see v5.1b's own shadow
            ge_add = Z[f"GE|{p}"]; Z[f"GE|{p}"] = ge0; _, sh = A.shadow(A.mix(bn, p, list(A.FAMS))[0], p); Z[f"GE|{p}"] = ge_add
        else:
            _, sh = A.shadow(mix, p)
        assert abs(a - (1 / 16 if "R" in arm else 1 / 15)) < 1e-12, (arm, a)
        out0 = ns["route4"]({p: mix}, {p: sh}, a, oof, {p: A.members(bn, p)["c1"]}, "max", 10)[0]
        rt = {s.split("/")[-1]: (int(np.flatnonzero(r > 0)[0]), K.NM[(q, s)][0][int(j[np.flatnonzero(r > 0)[0]])][-3:], r.copy()) for (q, s), (r, j) in K.REC.items()}
        out = K.staged(out0)[0]
        if LABEL_RT and arm == "base" and bn == "B2_pool":  # s190: the routed v5.1b outputs as emitted, back in recording layout
            LABRT[0] = out[p][np.argsort(Pp)]
    finally:
        A.RR.FOLDS = folds0; Z[f"GE|{p}"] = ge0
        for k in [k for k in A.EXTRA if k.startswith("rt_")]:
            del A.EXTRA[k]
    return SC.person(out[p], p), rt


OOF = {bn: oof_rec(bn) for bn in BASES}; R, RT = {}, {}
for order in ORDERS:
    P = perm(order); Pp = P[p]; inp = inputs(order)
    for c in (ns["_BIO"], A._H, K.STREAMS, K.REC):  # session-keyed caches depend on the delivery order only
        c.clear()
    S.apply(P); K.RANK["cur"] = None if order == "rec" else {q: K._rank(K.IDX[q][P[q]]) for q in PR}
    if RT_MEMBERS and order != "rec":
        apply_members(member_deltas(order), Pp); print(f"  {order}: returning people's core1, Hilbert, teacher, pc128, GE and frozen SessionBN re-streamed (s214)", flush=True)
    for bn in BASES:
        for arm in ARMS:
            R[order, bn, arm], RT[order, bn, arm] = system(bn, arm, Pp, OOF[bn], inp)
        key = f"rec|0|{bn}|{p}" if order == "rec" else f"{order[:-1]}|{order[-1]}|{bn}|{p}"
        if key in REF.files and not (RT_MEMBERS and order != "rec"):  # shufpkg keeps members as recorded: comparable only then
            assert np.allclose(R[order, bn, "base"], REF[key], atol=1e-9), f"{order} {bn}: base {R[order, bn, 'base']} != shufpkg {REF[key]}"
        elif bn == BASES[0]:
            print(f"  {order}: no shufpkg reference for this order; routed v5.1b is computed by the same code but not checked against one", flush=True)
        b = R[order, bn, "base"]
        print(f"  {order} {bn}: base P04 {b[0]:.2f} P10 {b[1]:.2f} ({'= shufpkg' if key in REF.files else 'no shufpkg reference'}) | " + " ".join(f"{k} {(R[order, bn, k] - b).mean():+.2f} ({R[order, bn, k][0] - b[0]:+.2f}, {R[order, bn, k][1] - b[1]:+.2f})" for k in ARMS[1:]), flush=True)
        for k in ARMS[1:]:
            r0, r1 = RT[order, bn, "base"], RT[order, bn, k]
            same_set = set(r0) == set(r1); first = {s: (r0[s][0], r1[s][0]) for s in r0 if s in r1 and r0[s][:2] != r1[s][:2]}
            ramp = sum(not np.array_equal(r0[s][2], r1[s][2]) for s in r0 if s in r1)
            if not same_set or first or ramp:
                print(f"    routing changes in {k}: sessions {sorted(r0)} -> {sorted(r1)}; first window/person changed {first}; ramp differs in {ramp} sessions", flush=True)
    if LABEL_RT:
        assert order != "rec", "LABEL_RT needs a delivery order (ttt18 has no recording-order form)"
        L = np.load(f"{REPO}/runs/labels/chain_v51b_{order}.npz"); assert (L["idx_seen"] == IDX).all()
        y = Z[f"y|{p}"].astype(int)[np.argsort(Pp)]; lab = LABRT[0]
        print(f"  labels {order}: routed v5.1b acc {np.mean(lab.argmax(1) == y):.4f} | chain_final seen {np.mean(L['seen_session'].argmax(1) == y):.4f}", flush=True)
        np.savez(f"{REPO}/runs/labels/chain_rt_{order}.npz", all_loso=L["all_loso"], seen_session=lab, idx_seen=L["idx_seen"]); print("written chain_rt_" + order, flush=True)
        continue
    print(f"  {order}: routed sessions (base, first window, person) " + "; ".join(f"{s} w{v[0]} {v[1]}" for s, v in RT[order, BASES[0], 'base'].items()), flush=True)
S.apply({q: np.arange(S.N[q]) for q in PR}); K.RANK["cur"] = None
if LABEL_RT:
    sys.exit(0)
print(f"\nRETV6: change against routed v5.1b in the same order, mean over {len(BASES)} bases: mean (P04, P10) [per-base range of the mean]")
for order in ORDERS:
    for k in ARMS[1:]:
        d = np.array([R[order, bn, k] - R[order, bn, "base"] for bn in BASES]); m = d.mean(0)
        print(f"  {order:8s} {k:4s} {m.mean():+.2f} (P04 {m[0]:+.2f}, P10 {m[1]:+.2f}) [{d.mean(1).min():+.2f}, {d.mean(1).max():+.2f}]", flush=True)
sh_ = [o for o in ORDERS if o != "rec"]
for k in ARMS[1:] if sh_ else []:
    m = np.mean([R[o, bn, k] - R[o, bn, "base"] for o in sh_ for bn in BASES], 0)
    print(f"  shuffled mean ({','.join(sh_)}) {k:4s} {m.mean():+.2f} (P04 {m[0]:+.2f}, P10 {m[1]:+.2f})", flush=True)
# one file per setting and order set (s202): runs with other settings no longer overwrite each other's results
TAGS = "_".join(x for x in ("p0" if RT_MEMBERS else "", "" if len(BASES) == 6 else "smoke", f"g{int(GATE_V51B)}", FT_FORM, FT_SFX, p, "-".join(ORDERS)) if x)  # p0: s214 fixes
os.makedirs(f"{REPO}/runs/retv6", exist_ok=True); np.savez(f"{REPO}/runs/retv6/{TAGS}.npz", **{f"{o}|{bn}|{k}": v for (o, bn, k), v in R.items()}); print(f"written runs/retv6/{TAGS}.npz", flush=True)
