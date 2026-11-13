"""TTT8 (13 Nov, ledger s131): follow-ups of cum10 (s108's counting arm: warm start, each refit continues from the
previous one). ttt6.py's machinery with other variants; same nets, folds, augmentation and chain labels.
  cum10    the s108 arm again (in-run reference)        cost 0.5 of fixA
  cum20    warm start, 20 steps every 10 windows          cost 1
  cum10e5  warm start, 10 steps every 5 windows           cost 1
  cum5e5   warm start, 5 steps every 5 windows            cost 0.5
  cum10lr  warm start, 10 steps, learning rate 1e-3       cost 0.5
  cum10c2  warm start, 10 steps, consistency weight 2     cost 0.5
Saved to runs/diag/ttt/probs8 as ttt8_<arch>_<variant>-<proto>-s<seed>.npz, plus base.
Usage (Leonardo): python studies/13nov/ttt8.py TASK (arch = ORDER[TASK % 4], seed = TASK // 4) | smoke"""
import importlib.util, os
from pathlib import Path
WS = Path(os.environ.get("NI26_WS", Path(__file__).resolve().parents[2]))
_spec = importlib.util.spec_from_file_location("ttt6", WS / "studies/13nov/ttt6.py"); T6 = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(T6)
T6.TAG, T6.OUT, T6.SMOKE_OUT = "ttt8", WS / "runs/diag/ttt/probs8", "smoke8"
# variant -> (steps per refit, refit every, lr, scope, consistency weight, label form, warm start, views)
T6.VARS = {"cum10": (10, 10, 5e-4, "all", 1.0, "hard", True, 1), "cum20": (20, 10, 5e-4, "all", 1.0, "hard", True, 1),
           "cum10e5": (10, 5, 5e-4, "all", 1.0, "hard", True, 1), "cum5e5": (5, 5, 5e-4, "all", 1.0, "hard", True, 1),
           "cum10lr": (10, 10, 1e-3, "all", 1.0, "hard", True, 1), "cum10c2": (10, 10, 5e-4, "all", 2.0, "hard", True, 1)}
assert T6.run.__globals__["VARS"] is T6.VARS and T6.run.__globals__["TAG"] == "ttt8", "override did not reach ttt6's functions"

if __name__ == "__main__":
    T6.main()
