"""TTT15 (13 Nov, ledger s154): fine-tuning that keeps the hard windows without trusting their labels: hard labels on
the confident half of the session prefix (chain confidence above the prefix median), entropy minimisation on the
low-confidence half (CONFHALF, s145 C, showed that dropping them hurts). Warm start, 10 steps every 10 windows (cum10
schedule), same nets as ttt6 seeds 0-5, so it is scored as a change to the shipped members against v5.1b
(pkg.py FT_DIR=runs/diag/ttt/probs15 FT_PFX=ttt15 FT_VAR=hent). Usage (Leonardo): python studies/13nov/ttt15.py TASK | smoke"""
import importlib.util, os
from pathlib import Path
WS = Path(os.environ.get("NI26_WS", Path(__file__).resolve().parents[2]))
assert "CHAIN_NPZ" not in os.environ, "TTT15 uses the v3 chain labels"
_spec = importlib.util.spec_from_file_location("ttt6", WS / "studies/13nov/ttt6.py"); T6 = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(T6)
T6.TAG, T6.OUT, T6.SMOKE_OUT = "ttt15", WS / "runs/diag/ttt/probs15", "smoke15"
T6.VARS = {"hent": (10, 10, 5e-4, "all", 1.0, "hent", True, 1)}
assert T6.stream.__globals__["VARS"] is T6.VARS and T6.run.__globals__["TAG"] == "ttt15"

if __name__ == "__main__":
    T6.main()
