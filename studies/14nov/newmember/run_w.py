"""NEWMEMBER chain (ledger s272): shufcombo (run unedited via runpy, env as the v6.6 chain) with a new net family as a member slot.
The new slot is the shuffled-delivery base stream itself: mean over the base's seeds of ttt18<order>_<fam>_base-all_loso-s<seed>.npz
(recording layout, streamed in the delivery order), no recording-order delta, no fine-tuning. Weight: ARM=A the weight of the replaced
fbraw slot (fbraw removed, net count equal), ARM=B the mean slot weight (added), ARM=Z weight 0 (zero-change check: the mix must equal
the shipped mix bitwise, asserted at every call). ARM unset or ctl: shufcombo unchanged (the control). Recording-order calls (P0rec,
V6Frec, P = identity) never see the new slot. The new slot also enters P0s/P0r/V6r (the package's inner v5.1b gate is built from the
same members). New people only (seen_session has no new-family files; its column stays NaN and the run asserts finite output).
Env: ARM=ctl|A|B|Z  NEWSLOT=eegn | eegn,conf (A: one family)  NM_TAG (default ttt18<order>)  NM_SEEDS (default 0,1,2,3,4,5)
     NM_DIR (default probs18 dir of the order)  NM_NAME (output name, default nm<ARM><fams>; ctl: ctl)
Output: runs/chainout/nm_<NAME>_<order>.npz (V6Fr/P0r/V6r per window, SAVE_OUT) and runs/newmember/<NAME>_shufcombo_*.npz.
Usage: ARM=A NEWSLOT=eegn BASES=... python studies/14nov/newmember/run.py ORDER   (e.g. within0)"""
import os, runpy, sys
import numpy as np
REPO = os.path.abspath(__file__ + "/../../../.."); O = sys.argv[1]; SC_ = O.rstrip("0123456789"); PS = O[len(SC_):]
ARM = os.environ.get("ARM", "ctl"); FAMS_ = [f for f in os.environ.get("NEWSLOT", "").split(",") if f]
SLOTS_ = ("fb", "fbl", "sinc", "fbraw"); DROP = os.environ.get("DROP"); HALF = os.environ.get("HALF")
assert ARM in ("ctl", "A", "B", "Z", "D", "H"), ARM
assert (ARM in ("ctl", "D", "H")) == (not FAMS_), f"ARM={ARM} NEWSLOT={FAMS_}: ctl, D and H take no NEWSLOT, every other arm needs one"
assert (ARM == "D") == (DROP is not None) and (DROP is None or DROP in SLOTS_ + ("none",)), f"ARM={ARM} DROP={DROP}: D needs DROP in {SLOTS_ + ('none',)}, no other arm takes it"
assert (ARM == "H") == (HALF is not None) and (HALF is None or HALF in SLOTS_), f"ARM={ARM} HALF={HALF}: H needs HALF in {SLOTS_}, no other arm takes it"
assert ARM == "H" or "HALF" not in os.environ, "HALF is set outside ARM=H (shufcombo would apply it)"
NEWW = float(os.environ.get("NEWW", "1.0")); assert ARM in ("A", "B", "ctl") and NEWW >= 0, "run_w.py: ARM A, B or ctl only (s275)"
assert "NEWW" in os.environ or ARM == "ctl", "set NEWW explicitly (s275)"
assert ARM != "A" or len(FAMS_) == 1, "arm A swaps one family for fbraw"
assert ARM != "B" or len(FAMS_) >= 2, "arm B adds two or more families"
assert ARM == "ctl" or f"_w{os.environ['NEWW']}" in (os.environ.get("NM_NAME") or ""), "NM_NAME must end in _w<NEWW> so outputs never collide"
REFIT = {k: float(v) for k, v in (kv.split("=") for kv in os.environ.get("REFIT", "").split(",") if kv)}  # s289: stage weights refit: <family|sc_slot>=<factor>
assert not REFIT or ARM in ("A", "B"), "REFIT needs ARM A or B (s289)"
assert all(k in ("core1", "hilb", "tf", "shallow", "teacher", "sc_fb", "sc_fbl", "sc_sinc") and v > 0 for k, v in REFIT.items()), f"unknown REFIT key in {REFIT}"
assert not REFIT or all(k in (os.environ.get("NM_NAME") or "") for k in REFIT), "NM_NAME must name every REFIT key so outputs never collide"
NAME = os.environ.get("NM_NAME") or ("ctl" if ARM == "ctl" else "nm" + ARM + (DROP or HALF or "".join(FAMS_))); NM_SEEDS = [int(s) for s in os.environ.get("NM_SEEDS", "0,1,2,3,4,5").split(",")]


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
NM_DIR = REPO + "/" + (E.get("NM_DIR") or where_file("ttt/probs18", f"{E.get('NM_TAG', 'ttt18' + O)}_{FAMS_[0]}_base-all_loso*")) if FAMS_ else None
TAG = E.get("NM_TAG", f"ttt18{O}")
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
if ARM in ("A", "B"):
    _mix = A.mix

    def mix(bn, p, fams, extra=(), *a, **k):
        m0 = _mix(bn, p, fams, extra, *a, **k)
        P = CUR["P"]
        if p != p0 or (P == np.arange(N)).all():  # recording-order calls and other protocols: the shipped system
            return m0
        if ARM == "D":  # s273: the dropped slot's weight is 0 in every re-streamed call, the others unchanged
            ex = [(n, 0.0 if n == f"sc_{DROP}" else w) for n, w in extra]; names = [n for n, _ in extra]
            assert names.count("sc_fbraw") == 1 and all(f"sc_{x}" in names for x in SLOTS_), names
            m = _mix(bn, p, fams, ex, *a, **k); d = float(np.abs(m[0] - m0[0]).max()); LOG.append((bn, ARM, 0.0 if DROP != "none" else 1.0, d))
            if DROP == "none":
                assert ex == extra and np.array_equal(m[0], m0[0]) and m[1] == m0[1], "DROP=none changed the mix"
            else:
                assert dict(ex)[f"sc_{DROP}"] == 0.0 and sum(w == 0.0 for _, w in ex) == 1 and dict(extra)[f"sc_{DROP}"] != 0.0, "dropped weight is not the only zero"
                assert d > 0, f"dropping {DROP} does not change the mix"
            return m
        extra = list(extra); sw = {n: w for n, w in extra if n in ("sc_fbraw", "sc_fb", "sc_sinc", "sc_fbl")}
        assert set(sw) == {"sc_fbraw", "sc_fb", "sc_sinc", "sc_fbl"} and sum(n == "sc_fbraw" for n, _ in extra) == 1, [n for n, _ in extra]
        if ARM == "A":
            n0 = len(extra); w = sw["sc_fbraw"]; base_w = w; extra = [(n, x) for n, x in extra if n != "sc_fbraw"]
            assert all(n != "sc_fbraw" for n, _ in extra) and len(extra) == n0 - 1, "the fbraw slot must be absent in the swap arm"
        else:
            base_w = float(np.mean(list(sw.values())))
        w = NEWW * base_w
        for fam in FAMS_:
            A.EXTRA[f"nm_{fam}"] = {p0: stream(fam, bn)[P], OTHER: np.full((S.N[OTHER], 3), np.nan)}; extra.append((f"nm_{fam}", w))
        m_nr = _mix(bn, p, fams, extra, *a, **k)  # arm without the s289 refit
        if REFIT:
            extra = [(n, x * REFIT.get(n, 1.0)) for n, x in extra]; assert "wscale" not in k, "wscale already set"
            k = dict(k, wscale={f: REFIT[f] for f in REFIT if f in A.FAM})
        m = _mix(bn, p, fams, extra, *a, **k); d = float(np.abs(m[0] - m0[0]).max()); LOG.append((bn, ARM, w, d))
        if REFIT:  # silent no-op guard: a non-unit factor must move the mix, all-unit must not; the new slot is not rescaled
            nu = any(v != 1.0 for v in REFIT.values()); assert (not np.array_equal(m[0], m_nr[0])) == nu, f"REFIT {REFIT} {'did not move' if nu else 'moved'} the mix"
            assert all(f in fams or f.startswith("sc_") for f in REFIT), f"REFIT family not in the chain: {REFIT} vs {fams}"
        assert abs(dict(extra)[f"nm_{FAMS_[0]}"] - NEWW * base_w) < 1e-12 and all(abs(x - NEWW * base_w) < 1e-12 for n, x in extra if isinstance(n, str) and n.startswith("nm_")), "new slot weight is not NEWW x base"
        if NEWW == 0:
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
print(f"NEWMEMBER {O} NEWW {NEWW} ARM {ARM} DROP {DROP} HALF {HALF} NEWSLOT {FAMS_} seeds {NM_SEEDS} tag {TAG} name {NAME}", flush=True)
runpy.run_path(f"{REPO}/studies/13nov/shufcombo.py", run_name="__main__")
if ARM in ("A", "B"):
    assert LOG, "the new slot was never mixed in (no delivery-order chain call)"
    print(f"NEWMEMBER mix calls {len(LOG)} slot weight {sorted({round(x[2], 4) for x in LOG})} max |mix - shipped mix| {max(x[3] for x in LOG):.4f} (zero for ARM=Z: asserted)", flush=True)
