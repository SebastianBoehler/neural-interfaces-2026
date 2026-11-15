"""s276 on the ten-person routed proxy (studies/14nov/pview/proxy_routed.py, run through a patched in-memory copy; the file is not edited): DROP = slot weights 0
in every A.mix call (the proxy's mix, as run_c.py / ret_c.py), HALF = for bases B2_pool and B2_s345 the slot rows are B2_s012's (members are as recorded in the proxy:
no FR/FT deltas, so halving the seed pool is the slot swap). Personal heads (ps1_heads on session-1 PCs), the own-space stage and the TSEM prior do not read the slots;
only R3.slot_mix blends the head into the mix at share a = the mix's core1 share, which a drop raises (the same renormalisation a removed net gives the package).
Control = DROP=none HALF=none: proxy_routed's own tail0 == tproxy zero-change assert stays on; with an arm it, and the a = 1/15, 1/16 asserts, are replaced by the arm's asserts.
Output runs/pview/routed262_<NM_NAME>_<orders>.npz (key order|base|arm -> (10,) online balanced accuracy per person).
Usage: DROP=fb,fbraw HALF=none NM_NAME=C1 ORDERS=within0 ARMS=tail0,ERT python studies/14nov/newmember/proxy_c.py"""
import os, sys
import numpy as np
REPO = os.path.abspath(__file__ + "/../../../.."); NAME = os.environ["NM_NAME"]; SLOT = {"fbraw": 5, "fb": 6, "sinc": 7, "fbl": 8}


def lst(k):
    v = os.environ.pop(k); r = [] if v == "none" else v.split(","); assert len(set(r)) == len(r) and set(r) <= set(SLOT), f"{k}={v}"; return r


DROP, HALF = lst("DROP"), lst("HALF"); ACT = bool(DROP or HALF); os.environ["TAG"] = NAME + "_"
F = f"{REPO}/studies/14nov/pview/proxy_routed.py"; src = open(F).read(); LOG = {"drop": 0, "half": 0}
REP = [("assert abs(a0 - 1 / 15) < 1e-12; ", ""), ("; assert abs(a - 1 / 16) < 1e-12", ""),
       ("                if arm == \"tail0\":\n", "                if arm == \"tail0\" and not ACT:\n")]
for o_, n_ in REP:
    assert src.count(o_) == 1, o_; src = src.replace(o_, n_)
sys.path[:0] = [REPO + "/studies/14nov/pview", REPO + "/studies/13nov", REPO + "/studies/12nov"]
import shuffle as S  # noqa: E402
A, ds, p0 = S.A, {SLOT[x] for x in DROP}, "all_loso"; _mix, _apply = A.mix, S.apply


def mix(bn, p, fams, extra=(), *a, **k):
    m0 = _mix(bn, p, fams, extra, *a, **k)
    if not ds:
        return m0
    moved = [f for f in fams if any(kk in ds for kk, _ in A.FAM[f])]; assert {kk for f in moved for kk, _ in A.FAM[f]} >= ds, (fams, ds)
    ex = [(kk, 0.0 if kk in ds else w) for f in moved for kk, w in A.FAM[f]] + list(extra)
    m = _mix(bn, p, [f for f in fams if f not in moved], ex, *a, **k)
    assert sorted(kk for kk, w in ex if w == 0.0) == sorted(ds) and np.abs(m[0] - m0[0]).max() > 0, "drop is not exactly the zero weights / does not change the mix"
    LOG["drop"] += 1; return m


def apply(P):
    _apply(P); Z = A.L.Z
    for bn in ("B2_pool", "B2_s345") if HALF else ():
        Ps = Z[f"{bn}|Ps|{p0}"].copy()
        for a in HALF:
            Ps[SLOT[a]] = Z[f"B2_s012|Ps|{p0}"][SLOT[a]]
        Z[f"{bn}|Ps|{p0}"] = Ps; LOG["half"] += 1


A.mix, S.apply = mix, apply
G_ = {"__name__": "__main__", "__file__": F, "ACT": ACT}
exec(compile(src, F, "exec"), G_)  # noqa: S102
assert (not DROP or LOG["drop"]) and (not HALF or LOG["half"]), LOG
print(f"PROXY-C {NAME} DROP {DROP} HALF {HALF}: drop mix calls {LOG['drop']}, halved applies {LOG['half']}", flush=True)
