"""TTT10 (13 Nov, ledger s139): the cheap head-only form done right (review 13 Nov: s108's "head" arm never trained
the output bias and ran at 5e-4). ttt6.py's machinery, scope head2 = SessionBN affine + the last two parameters.
  cum10    s108's recipe, in-run reference (full parameter set, warm start, 10 steps every 10 windows, 5e-4)
  head2    head2 scope, 20 steps every 10 windows, each refit from the frozen net, 2e-3 (the personal-head lr)
  head2c   head2 scope, warm start, 10 steps every 10 windows, 2e-3
  head2x   head2 scope, warm start, 20 steps every 10 windows, 5e-3
Saved to runs/diag/ttt/probs10 as ttt10_<arch>_<variant>-<proto>-s<seed>.npz (with the label file's hash).
Usage (Leonardo): python studies/13nov/ttt10.py TASK (arch = ORDER[TASK % 4], seed = TASK // 4) | smoke"""
import importlib.util, os
from pathlib import Path
WS = Path(os.environ.get("NI26_WS", Path(__file__).resolve().parents[2]))
assert "CHAIN_NPZ" not in os.environ, "TTT10 uses the v3 chain labels; a CHAIN_NPZ in the environment would change them silently"
_spec = importlib.util.spec_from_file_location("ttt6", WS / "studies/13nov/ttt6.py"); T6 = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(T6)
T6.TAG, T6.OUT, T6.SMOKE_OUT = "ttt10", WS / "runs/diag/ttt/probs10", "smoke10"
# variant -> (steps per refit, refit every, lr, scope, consistency weight, label form, warm start, views)
T6.VARS = {"cum10": (10, 10, 5e-4, "all", 1.0, "hard", True, 1), "head2": (20, 10, 2e-3, "head2", 1.0, "hard", False, 1),
           "head2c": (10, 10, 2e-3, "head2", 1.0, "hard", True, 1), "head2x": (20, 10, 5e-3, "head2", 1.0, "hard", True, 1)}
assert T6.run.__globals__["VARS"] is T6.VARS and T6.run.__globals__["TAG"] == "ttt10"

if __name__ == "__main__":
    T6.main()
