"""TTT18 (13 Nov, ledger s171): gC5 (the fine-tuning form that fits the run time, s120 addendum 3) re-streamed in a
non-chronological delivery order (runs/shuffle_perms.npz), same nets as ttt7 seeds 0-5; scored with shufcombo.py
(FT_KIND=gC5). Saved to runs/diag/ttt/probs18 as ttt18<SCHEME><PSEED>_<arch>_{base,gC5}-<proto>-s<seed>.npz.
CHAIN_NPZ=runs/labels/chain_v51b_<order>.npz: package-faithful pseudo-labels (tag suffix L, s178).
Usage (Leonardo): SCHEME=within PSEED=0 python studies/13nov/ttt18.py TASK | smoke"""
import importlib.util, os
from pathlib import Path
import numpy as np
WS = Path(os.environ.get("NI26_WS", Path(__file__).resolve().parents[2])); SCHEME = os.environ["SCHEME"]; PSEED = int(os.environ["PSEED"]); DRAW = int(os.environ.get("DRAW", 0))
LB = os.environ.get("CHAIN_NPZ")  # s178: v5.1b's labels as emitted in this order (shufcombo.py LABEL_OUT); must name the same order
LSFX = {f"chain_v51b_{SCHEME}{PSEED}.npz": "L", f"chain_v6r_{SCHEME}{PSEED}.npz": "R", f"chain_rt_{SCHEME}{PSEED}.npz": "T"}  # T (s190): returning rows labelled by routed v5.1b  # s187: R = labels from the v6 chain (eye + rest)
assert LB is None or os.path.basename(LB) in LSFX, f"label file {LB} is not this order's"
_spec = importlib.util.spec_from_file_location("ttt7", WS / "studies/13nov/ttt7.py"); T7 = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(T7)
P = np.load(WS / "runs/shuffle_perms.npz"); PERM = {}
for proto in ("all_loso", "seen_session"):
    order = P[f"{SCHEME}|{PSEED}|{proto}"]; rk = np.full(int(order.max()) + 10000, -1); rk[order] = np.arange(len(order)); PERM[proto] = rk
T7.TAG, T7.OUT, T7.SMOKE_OUT, T7.PERM, T7.DRAW = f"ttt18{SCHEME}{PSEED}" + (f"d{DRAW}" if DRAW else "") + (LSFX[os.path.basename(LB)] if LB else ""), WS / "runs/diag/ttt/probs18", f"smoke18{SCHEME}", PERM, DRAW  # DRAW > 0: fresh augmentation draws
T7.VARS = {"gC5": T7.VARS["gC5"]}
assert T7.run.__globals__["PERM"] is PERM and T7.stream.__globals__["DRAW"] == DRAW

if __name__ == "__main__":
    T7.main()
