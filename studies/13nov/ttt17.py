"""TTT17 (13 Nov, ledger s165): the SessionBN members and their fine-tuning (cum10) re-streamed in a non-chronological
delivery order, the same orders as studies/13nov/shuffle.py (runs/shuffle_perms.npz: SCHEME|SEED|proto -> global window
indices in delivery order). Causal alignment, SessionBN state and the fine-tuning schedule then follow that order.
Pseudo-labels: the v3 chain in recording order, or with CHAIN_NPZ=studies/13nov/chain_v51b_<order>.npz v5.1b as emitted in
that order (tag suffix L, s182).
Saved to runs/diag/ttt/probs17 as ttt17<SCHEME><SEED>_<arch>_{base,cum10}-<proto>-s<seed>.npz, keyed by window index.
Usage (Leonardo): SCHEME=full PSEED=0 python studies/13nov/ttt17.py TASK | smoke"""
import importlib.util, os
from pathlib import Path
import numpy as np
WS = Path(os.environ.get("NI26_WS", Path(__file__).resolve().parents[2])); SCHEME = os.environ["SCHEME"]; PSEED = int(os.environ["PSEED"])
LB = os.environ.get("CHAIN_NPZ")  # s182: v5.1b's labels as emitted in this order (shufcombo.py LABEL_OUT); must name the same order
assert LB is None or os.path.basename(LB) == f"chain_v51b_{SCHEME}{PSEED}.npz", f"label file {LB} is not this order's"
_spec = importlib.util.spec_from_file_location("ttt6", WS / "studies/13nov/ttt6.py"); T6 = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(T6)
P = np.load(WS / "runs/shuffle_perms.npz"); PERM = {}
for proto in ("all_loso", "seen_session"):
    order = P[f"{SCHEME}|{PSEED}|{proto}"]; rk = np.full(int(order.max()) + 10000, -1); rk[order] = np.arange(len(order)); PERM[proto] = rk
T6.TAG, T6.OUT, T6.SMOKE_OUT, T6.PERM = f"ttt17{SCHEME}{PSEED}" + ("L" if LB else ""), WS / "runs/diag/ttt/probs17", f"smoke17{SCHEME}", PERM
T6.VARS = {"cum10": (10, 10, 5e-4, "all", 1.0, "hard", True, 1)}
assert T6.run.__globals__["PERM"] is PERM and T6.run.__globals__["TAG"] == f"ttt17{SCHEME}{PSEED}" + ("L" if LB else "")

if __name__ == "__main__":
    T6.main()
