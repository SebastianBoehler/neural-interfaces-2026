"""s281 step 1 chain: shufcombo (v6.6 chain env as run.py/run_c.py) with extra base-only fb-family slots (s272/s275 form: the shuffled-delivery base
stream, mean over the base's seeds of ttt18<order>_<fam>_base-all_loso-s<k>.npz, no recording-order delta, no fine-tuning).
Env: ADD = comma list of families added at the mean slot weight of {fb, fbl, sinc, fbraw} (net count +len(ADD));
     SWAP = one family that replaces the sc_fb slot at the weight of sc_fb (net count equal; the replaced slot loses its own FR/FT stream);
     ZERO=1: the ADD slots enter at weight 0 and the mix must equal the shipped mix bitwise (zero-change check);
     ADD=none SWAP=none: shufcombo unchanged (the control, must equal runs/chainout/nm_ctl_<order>.npz bitwise, asserted by score);
     NM_NAME (required, output runs/chainout/nm_<NAME>_<order>.npz), NM_SEEDS (default 0-5).
Own copy of run.py (the earlier executors' files are not touched). Usage: ADD=fb_coral SWAP=none NM_NAME=S1 python studies/14nov/newmember/run_s.py ORDER"""
import os, runpy, sys
import numpy as np
REPO = os.path.abspath(__file__ + "/../../../.."); O = sys.argv[1]; SC_ = O.rstrip("0123456789"); PS = O[len(SC_):]
E = os.environ; SLOTS_ = ("fb", "fbl", "sinc", "fbraw")
def lst(k):
    v = E[k]; return [] if v == "none" else v.split(",")
ADD, SW = lst("ADD"), lst("SWAP"); ZERO = E.get("ZERO") == "1"; NAME = E["NM_NAME"]; NM_SEEDS = [int(s) for s in E.get("NM_SEEDS", "0,1,2,3,4,5").split(",")]
assert len(SW) <= 1 and len(set(ADD)) == len(ADD) and not (ADD and SW) and all(f in ("fb_coral", "fb_am") for f in ADD + SW), (ADD, SW)
assert not ZERO or ADD, "ZERO needs ADD"
FAMS_ = ADD + SW; assert "ARM" not in E and "NEWSLOT" not in E and "DROP" not in E and "HALF" not in E


def pick(rel):  # the order's directory on this machine (Mac keeps part under runs/diag)
    hit = [d for d in (f"runs/{rel}", f"runs/diag/{rel}") if os.path.exists(f"{REPO}/{d}")]; return hit
def where(rel, order):
    hit = [d for d in pick(rel) if os.path.exists(f"{REPO}/{d}/{order}")]; assert hit, f"{rel}/{order} not found"; return hit[0]
def where_file(rel, pat):
    import glob; hit = [d for d in pick(rel) if glob.glob(f"{REPO}/{d}/{pat}")]; assert hit, f"{rel}/{pat} not found"; return hit[0]


E = os.environ
for k in ("SAVE_OUT", "RS_DIR", "CM_NPZ", "EXTRA_VIEW", "RUNOFF", "RUN_IDS", "NUIS", "TSEM_TGT", "ALLSEED", "EYEW", "REST_W", "EM_LAM", "TS_LAM", "BTS", "OCU_TS", "LABEL_OUT", "PROTO", "SYN_T", "OUTX", "TS_FILE", "OCU_SYN", "OCU_TX", "OCU_ORACLE"):
    assert k not in E, f"{k} is set: the run would not be the v6.6 chain"
ocu = where("ocushuf", O); t7 = where_file("ttt/probs7", "ttt7_fb_gC5*"); t18 = E.get("T17_DIR") or where_file("ttt/probs18", f"ttt18{O}L_fb_gC5*")
E.update(SCHEME=SC_, PSEED=PS, SCHEMES=SC_, PERMS=str(int(PS) + 1), STRICT="1", GATE_P0="1", OCU_DIR=f"{ocu}/{O}", RS_DIR=f"{REPO}/{where('restshuf', O)}", FT_V="gC5", REC_PFX="ttt7",
         T6_DIR=f"{REPO}/{t7}", T17_DIR=f"{REPO}/{t18}", FT_TAG=E.get("FT_TAG", f"ttt18{O}L"), SAVE_OUT=f"runs/chainout/nm_{NAME}_{O}.npz", PYTHONWARNINGS="ignore")
os.makedirs(f"{REPO}/runs/newmember", exist_ok=True); os.makedirs(f"{REPO}/runs/chainout", exist_ok=True)
sys.path[:0] = [REPO + "/studies/13nov"]
import shuffle as S  # noqa: E402  (shufcombo's `import shuffle as S` gets this module)
A, p0 = S.A, "all_loso"; OTHER = next(p for p in S.PR if p != p0); N = S.N[p0]
CUR, STREAM, LOG = {"P": np.arange(N)}, {}, []
NM_DIR = REPO + "/" + where_file("ttt/probs18", f"ttt18{O}_{FAMS_[0]}_base-all_loso*") if FAMS_ else None
TAG = f"ttt18{O}"
BS = {"B2_s0": [0], "B2_s1": [1], "B2_s2": [2], "B2_s012": [0, 1, 2], "B2_s345": [3, 4, 5], "B2_pool": [0, 1, 2, 3, 4, 5]}  # = shufcombo.BS


def norm(x):
    x = np.clip(x, 1e-6, None); return x / x.sum(1, keepdims=True)


def stream(fam, bn):
    """mean over the base's seeds of the family's base posteriors (recording layout); every seed of the base must exist."""
    ks = BS[bn]; miss = [k for k in ks if k not in NM_SEEDS]; assert not miss, f"{fam}: base {bn} needs seeds {miss}, NM_SEEDS={NM_SEEDS}"
    if (fam, bn) not in STREAM:
        zs = [np.load(f"{NM_DIR}/{TAG}_{fam}_base-{p0}-s{k}.npz") for k in ks]
        for z in zs:
            assert (z["index"] == np.arange(N)).all(), f"{fam}: index array differs from the replay order"
        assert len(zs) == len(ks); STREAM[fam, bn] = norm(np.mean([z["proba"] for z in zs], 0))
    return STREAM[fam, bn]


_apply = S.apply


def apply(P):  # shufcombo's chain() starts every call with S.apply({p: P}): remember the delivery order the stages see
    CUR["P"] = np.asarray(P[p0]); return _apply(P)


S.apply = apply
if FAMS_:
    _mix = A.mix

    def mix(bn, p, fams, extra=(), *a, **k):
        m0 = _mix(bn, p, fams, extra, *a, **k)
        P = CUR["P"]
        if p != p0 or (P == np.arange(N)).all():  # recording-order calls and other protocols: the shipped system
            return m0
        extra = list(extra); sw = {n: w for n, w in extra if n in ("sc_fbraw", "sc_fb", "sc_sinc", "sc_fbl")}
        assert set(sw) == {"sc_fbraw", "sc_fb", "sc_sinc", "sc_fbl"} and len(extra) == len(set(n for n, _ in extra)), [n for n, _ in extra]
        if SW:
            n0 = len(extra); w = sw["sc_fb"]; extra = [(n, x) for n, x in extra if n != "sc_fb"]
            assert all(n != "sc_fb" for n, _ in extra) and len(extra) == n0 - 1, "the fb slot must be absent in the swap arm"
        else:
            w = 0.0 if ZERO else float(np.mean(list(sw.values())))
        for fam in FAMS_:
            A.EXTRA[f"nm_{fam}"] = {p0: stream(fam, bn)[P], OTHER: np.full((S.N[OTHER], 3), np.nan)}; extra.append((f"nm_{fam}", w))
        m = _mix(bn, p, fams, extra, *a, **k); d = float(np.abs(m[0] - m0[0]).max()); LOG.append((bn, w, d))
        if ZERO:
            assert np.array_equal(m[0], m0[0]) and m[1] == m0[1], "zero-weight new slot changed the mix"
        else:
            assert d > 0, "the new slot does not change the mix"
        assert np.isfinite(m[0]).all(), "non-finite mix (the new slot's seen_session column leaked in?)"
        return m
    A.mix = mix
_savez = np.savez


def savez(path, *a, **k):  # shufcombo's summary file goes to runs/newmember so the shared runs/shufcombo_*.npz is never overwritten
    p = str(path)
    if os.path.basename(p).startswith("shufcombo_"):
        p = f"{REPO}/runs/newmember/{NAME}_{os.path.basename(p)}"
    return _savez(p, *a, **k)


np.savez = savez
print(f"S281 {O} ADD {ADD} SWAP {SW} ZERO {ZERO} seeds {NM_SEEDS} name {NAME}", flush=True)
runpy.run_path(f"{REPO}/studies/13nov/shufcombo.py", run_name="__main__")
if FAMS_:
    assert LOG, "the new slot was never mixed in (no delivery-order chain call)"
    print(f"S281 mix calls {len(LOG)} slot weight {sorted({round(x[1], 4) for x in LOG})} max |mix - shipped mix| {max(x[2] for x in LOG):.4f} (zero for ZERO=1: asserted)", flush=True)
