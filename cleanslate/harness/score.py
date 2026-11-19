"""Collect stream outputs into official cell scores (labels are opened only here).

python score.py RUN_DIR [--data PATH]    RUN_DIR holds pred_<mode>_<person>.npz (mode new|ret) from stream.py.
Writes RUN_DIR/cells.npz (one record per cell) and RUN_DIR/scores.json (cell mean per mode/order, per-person means).
"""
import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import data as D  # noqa: E402


def collect(run, d):
    cells, pred_bank = [], {}
    for f in sorted(Path(run).glob("pred_*_*.npz")):
        mode = f.name.split("_")[1]
        with np.load(f, allow_pickle=False) as z:
            for order in sorted({int(m.group(1)) for k in z.files if (m := re.fullmatch(r"rows_(\d+)_\w+", k))}):
                pred = np.full(len(d["y"]), -1, np.int8)
                mask = np.zeros(len(d["y"]), bool)
                for k in z.files:
                    m = re.fullmatch(rf"rows_{order}_(\w+)", k)
                    if m:
                        rows = z[k]
                        assert len(set(rows.tolist())) == len(rows)
                        pred[rows] = z[f"labels_{order}_{m.group(1)}"]
                        mask[rows] = True
                cells += D.cell_table(d, pred, mask, order, mode)
                pred_bank[f"{mode}|{order}|{f.stem.split('_')[2]}"] = pred
    return cells, pred_bank


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--data", default=None)
    a = ap.parse_args()
    d = D.load(a.data)
    jobs = Path(a.run) / "jobs.tsv"
    if jobs.exists():  # fail loudly on any missing fold
        for line in jobs.read_text().splitlines():
            mode, held = line.split("\t")[:2]
            assert (Path(a.run) / f"pred_{mode}_{held}.npz").exists(), f"fold {mode} {held} has no predictions"
    cells, pred_bank = collect(a.run, d)
    assert cells, "no cells"
    cols = {k: np.array([c[k] for c in cells]) for k in cells[0]}
    np.savez(Path(a.run) / "cells.npz", **cols)
    np.savez_compressed(Path(a.run) / "predictions.npz", **pred_bank)
    out = {}
    for mode in sorted(set(cols["mode"])):
        for order in sorted(set(cols["order"][cols["mode"] == mode])):
            m = (cols["mode"] == mode) & (cols["order"] == order)
            persons = sorted(set(cols["person"][m]))
            out[f"{mode}|{order}"] = dict(cell_mean=float(cols["ba"][m].mean()), n_cells=int(m.sum()),
                                         by_person={p: float(cols["ba"][m & (cols["person"] == p)].mean()) for p in persons})
    json.dump(out, open(Path(a.run) / "scores.json", "w"), indent=1)
    for k, v in out.items():
        print(f"{k}: cell mean {v['cell_mean']:.3f} % over {v['n_cells']} cells")


if __name__ == "__main__":
    main()
