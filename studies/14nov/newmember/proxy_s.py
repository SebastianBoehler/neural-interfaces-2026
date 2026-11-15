"""s281 step 1 on the ten-person routed proxy (studies/14nov/pview/proxy_routed.py through a patched in-memory copy, as proxy_c.py): the proxy's members are as
recorded (no FR/FT deltas), so the new slots are exactly run_s.py's base-only slots: ADD = families added at the mean weight of the four slots 5-8 of alib.FAM (net
count +len), SWAP = one family replacing slot 6 (fb; its weight set to 0, the new slot at that weight); ZERO=1: ADD slots at weight 0, proxy_routed's own asserts
stay on (tail0 == tproxy, a = 1/15, 1/16) and every mix must equal the unmodified mix. Base streams: runs/ttt/probs18/ttt18<order>_<fam>_base-all_loso-s<k>.npz, mean over
the base's seeds, streamed in the order's delivery order. Output runs/pview/routed262_<NM_NAME>_<orders>.npz (key order|base|arm -> (10,)).
Usage: ADD=fb_coral SWAP=none NM_NAME=S1 ORDERS=within0 ARMS=tail0,ERT python studies/14nov/newmember/proxy_s.py"""
import os, sys
import numpy as np
REPO = os.path.abspath(__file__ + "/../../../.."); E = os.environ; NAME = E["NM_NAME"]; E["TAG"] = NAME + "_"
def lst(k):
    v = E.pop(k); return [] if v == "none" else v.split(",")
ADD, SW = lst("ADD"), lst("SWAP"); ZERO = E.pop("ZERO", "") == "1"; FAMS_ = ADD + SW; ACT = not ZERO
assert len(SW) <= 1 and not (ADD and SW) and FAMS_ and all(f in ("fb_coral", "fb_am") for f in FAMS_) and not (ZERO and SW), (ADD, SW, ZERO)
BS = {"B2_s0": [0], "B2_s1": [1], "B2_s2": [2], "B2_s012": [0, 1, 2], "B2_s345": [3, 4, 5], "B2_pool": [0, 1, 2, 3, 4, 5]}
F = f"{REPO}/studies/14nov/pview/proxy_routed.py"; src = open(F).read(); LOG = []
REP = [("assert abs(a0 - 1 / 15) < 1e-12; ", "assert ACT or abs(a0 - 1 / 15) < 1e-12; "), ("; assert abs(a - 1 / 16) < 1e-12", "; assert ACT or abs(a - 1 / 16) < 1e-12"),
       ("                if arm == \"tail0\":\n", "                if arm == \"tail0\" and not ACT:\n")]
for o_, n_ in REP:
    assert src.count(o_) == 1, o_; src = src.replace(o_, n_)
sys.path[:0] = [REPO + "/studies/14nov/pview", REPO + "/studies/13nov", REPO + "/studies/12nov"]
import shuffle as S  # noqa: E402
A, p0 = S.A, "all_loso"; _mix, _apply, _pf = A.mix, S.apply, S.perm_for; CUR = {}; SC_ = {}
OTHER = next(p for p in S.PR if p != p0)


def perm_for(sc, sd):
    CUR["order"] = f"{sc}{sd}"; return _pf(sc, sd)


def apply(P):
    CUR["P"] = np.asarray(P[p0]); return _apply(P)


def stream(fam, bn):
    key = (CUR["order"], fam, bn)
    if key not in SC_:
        d = f"{REPO}/runs/ttt/probs18"; zs = [np.load(f"{d}/ttt18{CUR['order']}_{fam}_base-{p0}-s{k}.npz") for k in BS[bn]]
        for z in zs:
            assert (z["index"] == np.arange(len(z["index"]))).all()
        x = np.clip(np.mean([z["proba"] for z in zs], 0), 1e-6, None); SC_[key] = x / x.sum(1, keepdims=True)
    return SC_[key]


def mix(bn, p, fams, extra=(), *a, **k):
    m0 = _mix(bn, p, fams, extra, *a, **k)
    if p != p0 or "P" not in CUR or (CUR["P"] == np.arange(len(CUR["P"]))).all():
        return m0
    sw = {kk: w for f in fams for kk, w in A.FAM[f] if kk in (5, 6, 7, 8)}; assert set(sw) == {5, 6, 7, 8}, sw
    ex = list(extra)
    w = 0.0 if ZERO else float(np.mean(list(sw.values())))
    if SW:
        w = sw[6]; moved = [f for f in fams if any(kk == 6 for kk, _ in A.FAM[f])]; assert len(moved) == 1
        ex = [(kk, 0.0 if kk == 6 else ww) for kk, ww in A.FAM[moved[0]]] + ex; fl = [f for f in fams if f != moved[0]]
    else:
        fl = list(fams)
    for fam in FAMS_:
        A.EXTRA[f"nm_{fam}"] = {p0: stream(fam, bn)[CUR["P"]], OTHER: np.full((S.N[OTHER], 3), np.nan)}; ex.append((f"nm_{fam}", w))
    m = _mix(bn, p, fl, ex, *a, **k); d = float(np.abs(m[0] - m0[0]).max()); LOG.append((bn, w, d))
    if ZERO:
        assert np.array_equal(m[0], m0[0]) and m[1] == m0[1], "zero-weight new slot changed the mix"
    else:
        assert d > 0 and np.isfinite(m[0]).all(), "the new slots do not change the mix / non-finite"
    return m


A.mix, S.apply, S.perm_for = mix, apply, perm_for
G_ = {"__name__": "__main__", "__file__": F, "ACT": ACT}
exec(compile(src, F, "exec"), G_)  # noqa: S102
assert LOG, "no delivery-order mix call saw the new slots"
print(f"PROXY-S {NAME} ADD {ADD} SWAP {SW} ZERO {ZERO}: mix calls {len(LOG)} weights {sorted({round(x[1], 4) for x in LOG})} max |mix change| {max(x[2] for x in LOG):.4f}", flush=True)
