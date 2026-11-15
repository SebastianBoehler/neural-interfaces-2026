"""s276 returning people (P04, P10; seen_session, sessions 2 and 3 in delivery orders): the ER chain of studies/14nov/pview/chain.py (run through a patched
in-memory copy; the file is not edited) with DROP (slot weights 0 in every delivery-order mix call of the routed system, as run_c.py) and HALF (the slot's
base and FR delta use seeds 0-2 for bases B2_pool and B2_s345, as run_c.py). Control = DROP=none HALF=none: all of chain.py's own checks stay on (ER equals
runs/chainout/ret_<order>.npz and the s214 reference); with a drop or halving those two reference checks and the a = 1/16 assertion are replaced by the
arm's own asserts. The recording-order shadow (OOF) is built outside the hooks. Output runs/chainout/ret_<NM_NAME>_<order>.npz (key <order>|<base>|ER).
Usage: DROP=fb,fbraw HALF=none NM_NAME=C1 python studies/14nov/newmember/ret_c.py ORDER"""
import os, sys
REPO = os.path.abspath(__file__ + "/../../../.."); O = sys.argv[1]; NAME = os.environ["NM_NAME"]; SLOT = {"fbraw": 5, "fb": 6, "sinc": 7, "fbl": 8}


def lst(k):
    v = os.environ.pop(k); r = [] if v == "none" else v.split(","); assert len(set(r)) == len(r) and set(r) <= set(SLOT), f"{k}={v}"; return r


DROP, HALF = lst("DROP"), lst("HALF"); ACT = bool(DROP or HALF)
os.environ.update(ORDERS=O, ARMS="ER", SAVE_OUT=f"runs/chainout/ret_{NAME}_{O}.npz", TAG=f"c_{NAME}_", PYTHONWARNINGS="ignore")
F = f"{REPO}/studies/14nov/pview/chain.py"; src = open(F).read(); LOG = {"drop": [], "half": []}
REP = [("def main():\n    t0 = time.time();", "def main():\n    HOOKS()\n    t0 = time.time();"),
       ("assert abs(a - (1 / 15 if arm == \"base\" else 1 / 16)) < 1e-12, (arm, a)", "assert ACT or abs(a - (1 / 15 if arm == \"base\" else 1 / 16)) < 1e-12, (arm, a)"),
       ("if SAVE_OUT and os.path.exists(f\"{D.REPO}/runs/chainout/ret_{order}.npz\"):", "if SAVE_OUT and not ACT and os.path.exists(f\"{D.REPO}/runs/chainout/ret_{order}.npz\"):"),
       ("            if REF is not None:\n", "            if REF is not None and not ACT:\n")]
for o_, n_ in REP:
    assert src.count(o_) == 1, o_; src = src.replace(o_, n_)
G_ = {"__name__": "__main__", "__file__": F, "ACT": ACT}


def HOOKS():
    A, R6, Z, p, S_ = G_["A"], G_["R6"], G_["Z"], G_["p"], G_["S"]; ds = {SLOT[x] for x in DROP}; on = [False]
    _mix, _sys, _am = A.mix, G_["system"], R6["apply_members"]

    def system(*a, **k):
        on[0] = True
        try:
            return _sys(*a, **k)
        finally:
            on[0] = False

    def mix(bn, pr, fams, extra=(), *a, **k):
        m0 = _mix(bn, pr, fams, extra, *a, **k)
        if not (on[0] and ds):
            return m0
        moved = [f for f in fams if any(kk in ds for kk, _ in A.FAM[f])]; assert {kk for f in moved for kk, _ in A.FAM[f]} >= ds, (fams, ds)
        ex = [(kk, 0.0 if kk in ds else w) for f in moved for kk, w in A.FAM[f]] + list(extra)
        m = _mix(bn, pr, [f for f in fams if f not in moved], ex, *a, **k)
        assert sorted(kk for kk, w in ex if w == 0.0) == sorted(ds) and np.abs(m[0] - m0[0]).max() > 0, "drop is not exactly the zero weights / does not change the mix"
        LOG["drop"].append((bn, float(m[1]), float(m0[1]))); return m

    def am(Dd, Pp):  # B2_pool / B2_s345 take the halved slots from B2_s012's own members (+ the mean FR delta of seeds 0-2), as shufcombo's HALF
        pre = {a: Z[f"B2_s012|Ps|{p}"][SLOT[a]].copy() for a in HALF}; _am(Dd, Pp)
        for bn in ("B2_pool", "B2_s345") if HALF else ():
            if bn not in G_["BASES"]:
                continue
            Ps = Z[f"{bn}|Ps|{p}"].copy()
            for a in HALF:
                x = np.clip(pre[a] + np.mean([Dd["fr"][a, k] for k in (0, 1, 2)], 0)[Pp], 1e-6, None); Ps[SLOT[a]] = x / x.sum(1, keepdims=True)
            Z[f"{bn}|Ps|{p}"] = Ps; LOG["half"].append((bn, tuple(HALF)))
    G_["system"], A.mix, R6["apply_members"] = system, mix, am


G_["HOOKS"] = HOOKS
import numpy as np  # noqa: E402
exec(compile(src, F, "exec"), G_)  # noqa: S102
if DROP:
    assert LOG["drop"], "no routed mix call saw the drop"
if HALF:
    assert {b for b, _ in LOG["half"]} == ({"B2_pool", "B2_s345"} & set(G_["BASES"])), LOG["half"]
print(f"RET-C {O} {NAME} DROP {DROP} HALF {HALF}: drop calls {len(LOG['drop'])} (core1 share a {sorted({round(x[1], 4) for x in LOG['drop']})[:3]} vs {sorted({round(x[2], 4) for x in LOG['drop']})[:2]}), halved bases {sorted({b for b, _ in LOG['half']})}", flush=True)
