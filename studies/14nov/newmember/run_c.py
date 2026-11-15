"""s276 compound arms C1-C4: shufcombo (v6.6 chain env as run.py, run through a patched in-memory copy, the file is not edited) with
DROP = comma list of slots whose weight is 0 in every delivery-order mix call (s273 D, as a list) and HALF = comma list of slots whose base and
FR/FT deltas use seeds 0-2 (B2_s012) on bases B2_pool, B2_s345 in delivery-order calls (s273 H, as a list). DROP=none / HALF=none: no effect
(DROP=none HALF=none must equal the control bitwise; scored). NM_NAME is required (output runs/chainout/nm_<NM_NAME>_<order>.npz).
Own copy of run.py (the s273 executor's file and the s275 executor's run_w.py are not touched). Control = runs/chainout/nm_ctl_<order>.npz.
Usage: DROP=fb,fbraw HALF=none NM_NAME=C1 python studies/14nov/newmember/run_c.py ORDER"""
import os, sys
import numpy as np
REPO = os.path.abspath(__file__ + "/../../../.."); O = sys.argv[1]; SC_ = O.rstrip("0123456789"); PS = O[len(SC_):]
SLOTS_ = ("fb", "fbl", "sinc", "fbraw"); NAME = os.environ["NM_NAME"]
def lst(k):
    v = os.environ[k]; r = [] if v == "none" else v.split(","); assert len(set(r)) == len(r) and set(r) <= set(SLOTS_), f"{k}={v}"; return r
DROP, HALF = lst("DROP"), lst("HALF"); assert "ARM" not in os.environ
def pick(rel):  # the order's directory on this machine (Mac keeps part under runs/diag)
    hit = [d for d in (f"runs/{rel}", f"runs/diag/{rel}") if os.path.exists(f"{REPO}/{d}")]; return hit
def where(rel, order):
    hit = [d for d in pick(rel) if os.path.exists(f"{REPO}/{d}/{order}")]; assert hit, f"{rel}/{order} not found"; return hit[0]
def where_file(rel, pat):
    import glob; hit = [d for d in pick(rel) if glob.glob(f"{REPO}/{d}/{pat}")]; assert hit, f"{rel}/{pat} not found"; return hit[0]


E = os.environ
E.pop('HALF'); E.pop('DROP')  # shufcombo's own HALF hook is replaced by the list form below
for k in ("SAVE_OUT", "RS_DIR", "CM_NPZ", "EXTRA_VIEW", "RUNOFF", "RUN_IDS", "NUIS", "TSEM_TGT", "ALLSEED", "EYEW", "REST_W", "EM_LAM", "TS_LAM", "BTS", "OCU_TS", "LABEL_OUT", "PROTO", "SYN_T", "OUTX", "TS_FILE", "OCU_SYN", "OCU_TX", "OCU_ORACLE"):
    assert k not in E, f"{k} is set: the run would not be the v6.6 chain"
ocu = where("ocushuf", O); t7 = where_file("ttt/probs7", "ttt7_fb_gC5*"); t18 = E.get("T17_DIR") or where_file("ttt/probs18", f"ttt18{O}L_fb_gC5*")
E.update(SCHEME=SC_, PSEED=PS, SCHEMES=SC_, PERMS=str(int(PS) + 1), STRICT="1", GATE_P0="1", OCU_DIR=f"{ocu}/{O}", RS_DIR=f"{REPO}/{where('restshuf', O)}", FT_V="gC5", REC_PFX="ttt7",
         T6_DIR=f"{REPO}/{t7}", T17_DIR=f"{REPO}/{t18}", FT_TAG=E.get("FT_TAG", f"ttt18{O}L"), SAVE_OUT=f"runs/chainout/nm_{NAME}_{O}.npz", PYTHONWARNINGS="ignore")
os.makedirs(f"{REPO}/runs/newmember", exist_ok=True); os.makedirs(f"{REPO}/runs/chainout", exist_ok=True)
sys.path[:0] = [REPO + "/studies/13nov"]
import shuffle as S  # noqa: E402  (shufcombo's `import shuffle as S` gets this module)
A, p0 = S.A, "all_loso"; OTHER = next(p for p in S.PR if p != p0); N = S.N[p0]
CUR, LOG = {"P": np.arange(N)}, []
BS = {"B2_s0": [0], "B2_s1": [1], "B2_s2": [2], "B2_s012": [0, 1, 2], "B2_s345": [3, 4, 5], "B2_pool": [0, 1, 2, 3, 4, 5]}


_apply = S.apply


def apply(P):
    CUR["P"] = np.asarray(P[p0]); return _apply(P)


S.apply = apply
_mix = A.mix


def mix(bn, p, fams, extra=(), *a, **k):
    m0 = _mix(bn, p, fams, extra, *a, **k); P = CUR["P"]
    if not DROP or p != p0 or (P == np.arange(N)).all():  # recording-order calls and other protocols: the shipped system
        return m0
    names = [n for n, _ in extra]; assert names.count("sc_fbraw") == 1 and all(f"sc_{x}" in names for x in SLOTS_), names
    ex = [(n, 0.0 if n in {f"sc_{x}" for x in DROP} else w) for n, w in extra]
    m = _mix(bn, p, fams, ex, *a, **k); d = float(np.abs(m[0] - m0[0]).max()); LOG.append((bn, d))
    assert sorted(n for n, w in ex if w == 0.0) == sorted(f"sc_{x}" for x in DROP) and all(dict(extra)[f"sc_{x}"] != 0.0 for x in DROP), "dropped weights are not exactly the zeros"
    assert d > 0, f"dropping {DROP} does not change the mix"
    return m


A.mix = mix
_savez = np.savez


def savez(path, *a, **k):
    p = str(path)
    if os.path.basename(p).startswith("shufcombo_"):
        p = f"{REPO}/runs/newmember/{NAME}_{os.path.basename(p)}"
    return _savez(p, *a, **k)


np.savez = savez
print(f"NEWMEMBER-C {O} DROP {DROP} HALF {HALF} name {NAME}", flush=True)
# shufcombo with the HALF hook generalised to a list of slots (patched in memory, each replacement asserted)
F13 = f"{REPO}/studies/13nov/shufcombo.py"; src = open(F13).read()
REP = [('HALF = os.environ.get("HALF") or None; assert HALF is None or HALF in SLOT, HALF', 'HALF = HALFL or None; HALFN = {SLOT[h] for h in HALFL}; HALFLOG = []'),
       ('s_ == SLOT.get(HALF)', 's_ in HALFN'),
       ('ks = BS[HB.get(bn, bn) if a == HALF else bn]; x = base[sl].copy()', 'ks = BS[HB.get(bn, bn) if a in HALFL else bn]; HALFLOG.append((a, bn, tuple(ks))); x = base[sl].copy()')]
for o_, n_ in REP:
    assert src.count(o_) == 1, o_; src = src.replace(o_, n_)
G_ = {"__name__": "__main__", "__file__": F13, "HALFL": HALF}
exec(compile(src, F13, "exec"), G_)  # noqa: S102
if DROP:
    assert LOG, "no delivery-order mix call saw the drop"
L_ = G_["HALFLOG"]; pool = ("B2_pool", "B2_s345")
for a in SLOTS_:  # halved slots use seeds 0-2 on the acting bases in some call; no other slot ever does
    seen = {tuple(ks) for (a2, bn, ks) in L_ if a2 == a and bn in pool}
    assert ((0, 1, 2) in seen) == (a in HALF), f"slot {a}: halved={a in HALF}, seed lists on pool/s345 {seen}"
print(f"NEWMEMBER-C done: drop mix calls {len(LOG)} max |mix change| {max([x[1] for x in LOG], default=0):.4f}; halved slots {HALF}: seeds (0,1,2) verified on B2_pool/B2_s345", flush=True)
