"""TTT9 (13 Nov, ledger s135): does fine-tuning gain more with better labels? cum10 and fixA (ttt6.py machinery), with
pseudo-labels from LAB = stack (the FT+OCU stack chain, runs/labels/chain_stack.npz) or true (one-hot true labels:
the oracle ceiling, never shippable). Saved to runs/diag/ttt/probs9_<LAB> as ttt9<LAB>_<arch>_<variant>-<proto>-s<seed>.npz.
Usage (Leonardo): LAB=stack|true python studies/13nov/ttt9.py TASK (arch = ORDER[TASK % 4], seed = TASK // 4) | smoke"""
import importlib.util, os
from pathlib import Path
WS = Path(os.environ.get("NI26_WS", Path(__file__).resolve().parents[2])); LAB = os.environ["LAB"]
assert LAB in ("stack", "true", "stack2", "indep", "agree", "truelow", "confhalf"), LAB  # stack2: round 3 (make_labels.py); indep/agree/truelow: TTT12, s142 (make_labels2.py)
os.environ["CHAIN_NPZ"] = str(WS / f"runs/labels/chain_{LAB}.npz"); assert Path(os.environ["CHAIN_NPZ"]).exists()
_spec = importlib.util.spec_from_file_location("ttt6", WS / "studies/13nov/ttt6.py"); T6 = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(T6)
import numpy as np  # noqa: E402
assert np.array_equal(T6.T5.CH["all_loso"], np.load(os.environ["CHAIN_NPZ"])["all_loso"]), "ttt5 did not load the chosen labels"
T6.TAG, T6.OUT, T6.SMOKE_OUT = f"ttt9{LAB}", WS / f"runs/diag/ttt/probs9_{LAB}", f"smoke9_{LAB}"
T6.VARS = {"cum10": (10, 10, 5e-4, "all", 1.0, "hard", True, 1), "fixA": (20, 10, 5e-4, "all", 1.0, "hard", False, 1)}
assert T6.run.__globals__["VARS"] is T6.VARS and T6.run.__globals__["TAG"] == f"ttt9{LAB}"

if __name__ == "__main__":
    T6.main()
