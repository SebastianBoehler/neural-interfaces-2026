"""s288 driver: registers the family 'tsm' (TSMNet, SPD-BN) in ttt5's tables at run time and runs ttt18's machinery unchanged
(so no shared file is edited): same folds, orders, base-only posteriors, file tag ttt18<SCHEME><PSEED>d<DRAW>_tsm_base-all_loso-s<seed>.npz.
Env as ttt18 (SCHEME, PSEED, DRAW, BASE_ONLY=1, PROTOS=all_loso) and FAMS=tsm. Usage: python -I studies/15nov/tsm288/ttt288.py TASK | smoke"""
import importlib.util, os, sys
from pathlib import Path
WS = Path(os.environ.get("NI26_WS", Path(__file__).resolve().parents[3]))
assert os.environ.get("FAMS") == "tsm" and os.environ.get("BASE_ONLY"), "s288 is the base-only 'tsm' family"
sys.path[:0] = [str(WS / "experiments"), str(WS / "scripts"), str(Path(__file__).parent)]
_s = importlib.util.spec_from_file_location("ttt18", WS / "studies/13nov/ttt18.py"); T18 = importlib.util.module_from_spec(_s); _s.loader.exec_module(T18)
T7 = T18.T7; T5 = T7.T5
import spd  # noqa: E402
assert "tsm" not in T5.ARCHS, "family name already taken"
T5.ARCHS["tsm"] = ("tsmnet_sbn", 20.0)  # (label only; make() is replaced below), test prior k 20 as sinc/fbraw
_make, _check_am = T5.make, T5.check_am
T5.make = lambda arch, d, seed: spd.make(d, seed) if arch == "tsm" else _make(arch, d, seed)


def check_am(arch, m, d, te, proto="all_loso"):  # hooked into ttt7.run right after fit (before set_k)
    _check_am(arch, m, d, te, proto)
    if arch == "tsm":
        spd.check_tsm(m, d, te, proto)


T5.check_am = check_am
assert T5.ORDER == ["tsm"] and T7.T5.make is T5.make and T7.T5.check_am is check_am, "family registration did not take"
if __name__ == "__main__":
    T7.main()
