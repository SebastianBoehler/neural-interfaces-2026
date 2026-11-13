"""TTT13 (13 Nov, ledger s152): fresh augmentation draws of cum10 on the same nets as ttt6 seeds 0-5 (which reproduce
the dev bases' own SessionBN slots for B2_s0/s1/s2: fb, fbl exactly, sinc to 0.016), so fine-tuning can be scored as a
change to the shipped members. DRAW = 1, 2, 3 (env), cum10 only. Saved to runs/diag/ttt/probs13 as
ttt13d<DRAW>_<arch>_cum10-<proto>-s<seed>.npz (+ base, which must equal ttt6's base).
Usage (Leonardo): DRAW=1 python studies/13nov/ttt13.py TASK (arch = ORDER[TASK % 4], seed = TASK // 4) | smoke"""
import importlib.util, os
from pathlib import Path
WS = Path(os.environ.get("NI26_WS", Path(__file__).resolve().parents[2])); DRAW = int(os.environ["DRAW"]); assert DRAW >= 1
assert "CHAIN_NPZ" not in os.environ, "TTT13 uses the v3 chain labels"
_spec = importlib.util.spec_from_file_location("ttt6", WS / "studies/13nov/ttt6.py"); T6 = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(T6)
T6.TAG, T6.OUT, T6.SMOKE_OUT, T6.DRAW = f"ttt13d{DRAW}", WS / "runs/diag/ttt/probs13", f"smoke13d{DRAW}", DRAW
T6.VARS = {"cum10": (10, 10, 5e-4, "all", 1.0, "hard", True, 1)}
assert T6.run.__globals__["DRAW"] == DRAW and T6.stream.__globals__["DRAW"] == DRAW and T6.run.__globals__["TAG"] == f"ttt13d{DRAW}"

if __name__ == "__main__":
    T6.main()
