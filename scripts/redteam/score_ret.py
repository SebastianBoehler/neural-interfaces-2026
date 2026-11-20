"""Score returning fold streams (P04/P10, visit 1 enrolled) of one or more arms against a reference arm.

python scripts/redteam/score_ret.py DATA.npz REF_DIR ARM_DIR [ARM_DIR ...]
Each directory holds pred_ret_P04.npz and pred_ret_P10.npz as written by cleanslate/harness/stream.py
(keys rows_<order>_<visit>, labels_<order>_<visit>). Metric: balanced accuracy per person x visit x interface cell,
equal mean over cells (every public cell holds all three classes, so this equals the official cell metric).
Prints absolute scores per person and visit, and the paired gain over REF per order (orders both arms hold).
Fails loudly when an arm delivered other rows than the reference."""
import sys
from pathlib import Path

import numpy as np

PEOPLE = ("P04", "P10")


def cells(data, pred):
    y, itf = data["y"], data["interface"]
    out = {}
    for key in pred.files:
        if not key.startswith("rows_"):
            continue
        _, order, visit = key.split("_")
        rows, lab = pred[key], pred[f"labels_{order}_{visit}"].astype(int)
        for face in ("Graz", "BrainHero"):
            m = itf[rows] == face
            out[int(order), visit, face] = (100 * float(np.mean([(lab[m][y[rows][m] == c] == c).mean() for c in range(3)])), rows)
    return out


def main(data_path, ref_dir, *arm_dirs):
    data = np.load(data_path, allow_pickle=True)
    ref = {p: cells(data, np.load(Path(ref_dir) / f"pred_ret_{p}.npz")) for p in PEOPLE}
    for p in PEOPLE:
        print(f"REF {Path(ref_dir).name} {p}: " + "  ".join(
            f"visit {v} {np.mean([s for (o, vv, f), (s, _) in ref[p].items() if vv == v]):.2f}" for v in ("2", "3")))
    for d in arm_dirs:
        gains = {}
        for p in PEOPLE:
            arm = cells(data, np.load(Path(d) / f"pred_ret_{p}.npz"))
            shared = sorted(set(arm) & set(ref[p]))
            if not shared:
                raise SystemExit(f"{d}: no order shared with the reference for {p}")
            for k in shared:
                if not np.array_equal(arm[k][1], ref[p][k][1]):
                    raise SystemExit(f"{d}: delivery differs from the reference at {p} {k}")
            orders = sorted({k[0] for k in shared})
            gains[p] = [float(np.mean([arm[k][0] - ref[p][k][0] for k in shared if k[0] == o])) for o in orders]
            print(f"{Path(d).name} {p}: absolute {np.mean([arm[k][0] for k in shared]):.2f}  gain {np.mean(gains[p]):+.2f}  "
                  f"per order {[round(g, 2) for g in gains[p]]}  orders {orders}")
        n = min(len(g) for g in gains.values())
        print(f"{Path(d).name} equal cells: gain per order {[round((gains['P04'][i] + gains['P10'][i]) / 2, 2) for i in range(n)]}")


if __name__ == "__main__":
    main(*sys.argv[1:])
