"""TTT16 (13 Nov, ledger s161): gC5 with a window cap, the fine-tuning trained on the last CAP windows only, so its
cost grows linearly with session length (s120 addendum 2: it grows with the square). CAP = 60 or 100 (env), DRAW 0-3
(env; 0 = ttt7's draw). Same nets as ttt7 seeds 0-5, scored as a change to the shipped members inside V6F against v5.1b.
Saved to runs/diag/ttt/probs16 as ttt16c<CAP>d<DRAW>_<arch>_gC5-<proto>-s<seed>.npz (+ base).
Usage (Leonardo): CAP=60 DRAW=0 python studies/13nov/ttt16.py TASK | smoke"""
import importlib.util, os
from pathlib import Path
WS = Path(os.environ.get("NI26_WS", Path(__file__).resolve().parents[2])); CAP = int(os.environ["CAP"]); DRAW = int(os.environ.get("DRAW", 0)); assert CAP > 0
assert "CHAIN_NPZ" not in os.environ, "TTT16 uses the v3 chain labels"
_spec = importlib.util.spec_from_file_location("ttt7", WS / "studies/13nov/ttt7.py"); T7 = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(T7)
T7.TAG, T7.OUT, T7.SMOKE_OUT, T7.DRAW, T7.CAP = f"ttt16c{CAP}d{DRAW}", WS / "runs/diag/ttt/probs16", f"smoke16c{CAP}", DRAW, CAP
T7.VARS = {"gC5": T7.VARS["gC5"]}
assert T7.stream.__globals__["CAP"] == CAP and T7.stream.__globals__["DRAW"] == DRAW and T7.run.__globals__["TAG"] == f"ttt16c{CAP}d{DRAW}"

if __name__ == "__main__":
    T7.main()
