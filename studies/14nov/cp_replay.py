"""s268 package form against the study form: lean/src66/cpers.py fed with the study's own inputs (data windows in delivery order,
the chain's raw posterior as p, its count-filter output as o) on chosen (order, base) cells of a draw; compared with the study's
pp file (k 20, cq50 / a2) and with score260's fusion. Prints per cell: windows, max |q - pp|, label agreement of the fused
output with the study's, refits. Exit 1 if q differs > 1e-6 or any fused label differs.
Usage: python -I studies/14nov/cp_replay.py DRAW ORDER:BASE [ORDER:BASE ...]   (DRAW d1|d2; e.g. d1 full0:0 within0:3)"""
import importlib.util, json, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[2]
sp = importlib.util.spec_from_file_location("cpers", ROOT / "lean/src66/cpers.py"); CP = importlib.util.module_from_spec(sp); sp.loader.exec_module(CP)
T0, R, CAP = 90, 120, 0.10
Z0 = {"v6_cp_fs": 120.0, "v6_cp_t0": 20, "v6_cp_k": 20, "v6_cp_w0": T0, "v6_cp_R": R, "v6_cp_cap": CAP, "v6_cp_q": 0.5,
      "v6_cp_bands": np.array([(4, 8), (8, 12), (12, 16), (16, 20), (20, 24), (24, 28), (28, 32), (32, 40)], float)}


def main():
    D, cells = sys.argv[1], [c.split(":") for c in sys.argv[2:]]
    Z = np.load(ROOT / f"runs/causalpers/chain{D}.npz"); X = np.load(ROOT / "data/proteus_windows.npz")["X"]
    sname = Z["sess"]  # row-level session id of the scoring layout (run.py asserts it matches the protocol sessions)
    ok, cp = True, CP.CausalPers(Z0)
    for o, b in cells:
        pz = np.load(ROOT / f"runs/causalpers/pp_{o}_b{b}_cq50_pa2_{D}.npz")["cq50|a2|k20"]; dl = Z[f"{o}|deliv"]
        raw, filt = Z[f"{o}|{b}|raw"], Z[f"{o}|{b}|filt"]; dq, nlab, nwin, nfus = 0.0, 0, 0, 0
        for s in dict.fromkeys(sname[dl]):
            rows = dl[sname[dl] == s]; cp.reset_state()
            for i, r in enumerate(rows):
                L = cp.step(X[r].astype(np.float64), raw[r], filt[r], False)
                w = CAP * np.clip((i - T0 + 1) / R, 0, 1)
                if i >= 20:
                    dq = max(dq, float(np.abs(cp.predict(cp.C[i]) - np.nan_to_num(pz[r], nan=1 / 3)).max())) if w > 0 else dq
                ref = np.log(np.clip(filt[r], 1e-6, 1)) + (w * np.log(np.clip(np.nan_to_num(pz[r], nan=1 / 3), 1e-6, 1)) if w > 0 else 0)
                nwin += 1; nlab += int(np.argmax(L) != np.argmax(ref)); nfus += int(w > 0)
            assert cp.log["err"] == 0, cp.log
        good = dq <= 1e-6 and nlab == 0; ok &= good
        print(json.dumps(dict(draw=D, order=o, base=int(b), windows=nwin, fused=nfus, max_q_diff=dq, label_disagree=nlab, ok=bool(good))), flush=True)
    print("RESULT", json.dumps(dict(ok=bool(ok))), flush=True); sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
