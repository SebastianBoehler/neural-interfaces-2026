"""s290 part 2 driver (R3 student width, R4 TSMNet learning rate): registers single-arm families in ttt5's tables at run time and runs ttt18's machinery unchanged
(no shared file is edited): same folds, orders, base-only posteriors; file tag ttt18<SCHEME><PSEED>d<DRAW>_<family>_base-all_loso-s<seed>.npz, the arm is in the family name.
  fbw2_dist2  fb student (FBCNet, n_filters_spat 32 -> 64 = width x2) distilled from the REVE teacher at T 2, weight 0.5 (fb_dist2 = width x1 is in the s283 grid)
  tsm_lr3e4 / tsm_lr3e3  TSMNet (s288 'tsm', default lr 1e-3) at lr 3e-4 / 3e-3
Env as ttt18 (SCHEME, PSEED, DRAW, BASE_ONLY=1, PROTOS=all_loso) and FAMS=<one family above>. Usage: python -I studies/15nov/s290/ttt290.py TASK | smoke (SMOKE_ARCH=<family> SMOKE_PROTO=all_loso)"""
import importlib.util, os, sys
from pathlib import Path
WS = Path(os.environ.get("NI26_WS", Path(__file__).resolve().parents[3]))
FAM = os.environ.get("FAMS")
FAMILIES = {"fbw2_dist2": dict(base="fb", dist=0.5, dist_T=2.0, kw=dict(n_filters_spat=64)), "tsm_lr3e4": dict(base="tsm", lr=3e-4), "tsm_lr3e3": dict(base="tsm", lr=3e-3)}
assert FAM in FAMILIES and os.environ.get("BASE_ONLY"), f"s290 part 2 families {list(FAMILIES)}, base-only; got FAMS={FAM}"
sys.path[:0] = [str(WS / "experiments"), str(WS / "scripts"), str(WS / "studies/15nov/tsm288")]
_s = importlib.util.spec_from_file_location("ttt18", WS / "studies/13nov/ttt18.py"); T18 = importlib.util.module_from_spec(_s); _s.loader.exec_module(T18)
T7 = T18.T7; T5 = T7.T5
import spd  # noqa: E402
SPEC = FAMILIES[FAM]
assert FAM not in T5.ARCHS and "tsm" not in T5.ARCHS, "family name already taken"
if SPEC["base"] == "fb":
    T5.ARM[FAM] = dict(base="fb", dist=SPEC["dist"], dist_T=SPEC["dist_T"])
T5.ARCHS[FAM] = T5.ARCHS["fb"] if SPEC["base"] == "fb" else ("tsmnet_sbn", 20.0)  # tsm: label only (make() is replaced below), k 20 as s288
_make, _check_am = T5.make, T5.check_am


def make(arch, d, seed):
    if arch != FAM:
        return _make(arch, d, seed)
    m = _make(arch, d, seed) if SPEC["base"] == "fb" else spd.make(d, seed)
    if SPEC["base"] == "fb":
        assert not m.net_kw, m.net_kw
        m.net_kw = dict(SPEC["kw"])
    else:
        assert m.lr == 1e-3, m.lr  # the s288 default this arm is measured against
        m.lr = SPEC["lr"]
    return m


def check_am(arch, m, d, te, proto="all_loso"):  # runs right after fit (ttt7): the arm really applied
    _check_am(arch, m, d, te, proto)
    if arch != FAM:
        return
    if SPEC["base"] == "fb":
        assert m.net.n_filters_spat == SPEC["kw"]["n_filters_spat"] and m.net_kw == SPEC["kw"], (type(m.net), m.net_kw)
        print(f"width ok {arch}: n_filters_spat {m.net.n_filters_spat}", flush=True)
    else:
        assert m.lr == SPEC["lr"], m.lr
        spd.check_tsm(m, d, te, proto); print(f"lr ok {arch}: {m.lr}", flush=True)


T5.make, T5.check_am = make, check_am
assert T5.ORDER == [FAM] and T7.T5.make is make and T7.T5.check_am is check_am, "family registration did not take"
if __name__ == "__main__":
    T7.main()
