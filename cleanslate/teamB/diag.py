"""Team B ceilings on end-of-session projections (diagnostics only; labels used for scoring and for the own-label arm).

python diag.py RUN_DIR   per cell, first order: enc_end (ensemble posterior under the final alignment), em_end (mixture
on the whole session, non-causal), own_ncm (leave-one-run-out nearest class centroid in the projection, the session's
own labels: the clusterability ceiling of this embedding).
"""
import importlib.util
import re
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "harness"))
import data as D  # noqa: E402

spec = importlib.util.spec_from_file_location("sub", HERE / "submission.py")
sub = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sub)


def main(run):
    d = D.load()
    tabs = {}
    for f in sorted(Path(run).glob("dev_*_*.npz")):
        mode = f.stem.split("_")[1]
        with np.load(f) as z:
            for k in z.files:
                m = re.fullmatch(r"Uend_(\d+)_(\w+)", k)
                if not m:
                    continue
                order, ses = m.groups()
                rows = z[f"rows_{order}_{ses}"]
                U, P = torch.from_numpy(z[k].astype(np.float32)), torch.from_numpy(z[f"Pend_{order}_{ses}"])
                y, runs = d["y"][rows], d["run"][rows]
                preds = dict(enc_end=P.argmax(1).numpy(), em_end=sub.transduce(P, U, sub.DEFAULTS)[1].argmax(1).numpy())
                ncm = np.zeros(len(rows), int)
                for r in np.unique(runs):
                    te = runs == r
                    mu = torch.stack([U[torch.from_numpy(~te & (y == c))].mean(0) for c in range(3)])
                    ncm[te] = (U[torch.from_numpy(te)] @ torch.nn.functional.normalize(mu, dim=1).T).argmax(1).numpy()
                preds["own_ncm"] = ncm
                for name, p in preds.items():
                    pred, mask = np.full(len(d["y"]), -1, np.int8), np.zeros(len(d["y"]), bool)
                    pred[rows], mask[rows] = p, True
                    tabs.setdefault((mode, name), []).extend(D.cell_table(d, pred, mask, int(order), mode))
    for (mode, name), cells in sorted(tabs.items()):
        people = sorted({c["person"] for c in cells})
        per = " ".join(f"{p[1:]}:{np.mean([c['ba'] for c in cells if c['person'] == p]):.0f}" for p in people)
        print(f"{mode:3} {name:8} {np.mean([c['ba'] for c in cells]):6.2f}  {per}")


if __name__ == "__main__":
    main(sys.argv[1])
