"""TTT14 (13 Nov, ledger s152): fresh augmentation draws of gC5 (warm start, 5 steps every 10 windows, GPU-drawn
augmentation; the cheapest fine-tuning form, s113) on the same nets as ttt7 seeds 0-5, scored as a change to the
shipped members (pkg.py SF / V6F with FT_VAR=gC5). DRAW = 1, 2, 3 (env). Saved to runs/diag/ttt/probs14 as
ttt14d<DRAW>_<arch>_gC5-<proto>-s<seed>.npz (+ base). Usage (Leonardo): DRAW=1 python studies/13nov/ttt14.py TASK | smoke"""
import importlib.util, os
from pathlib import Path
WS = Path(os.environ.get("NI26_WS", Path(__file__).resolve().parents[2])); DRAW = int(os.environ["DRAW"]); assert DRAW >= 1
assert "CHAIN_NPZ" not in os.environ, "TTT14 uses the v3 chain labels"
_spec = importlib.util.spec_from_file_location("ttt7", WS / "studies/13nov/ttt7.py"); T7 = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(T7)
T7.TAG, T7.OUT, T7.SMOKE_OUT, T7.DRAW = f"ttt14d{DRAW}", WS / "runs/diag/ttt/probs14", f"smoke14d{DRAW}", DRAW
T7.VARS = {"gC5": T7.VARS["gC5"]}
assert T7.stream.__globals__["DRAW"] == DRAW and T7.run.__globals__["TAG"] == f"ttt14d{DRAW}" and T7.stream.__globals__["VARS"] is T7.VARS

if __name__ == "__main__":
    T7.main()
