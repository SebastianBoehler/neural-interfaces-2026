"""Team B collector: python collect.py RUN_DIR [--ref predictions.npz]   (RUN_DIR holds dev_<mode>_<person>.npz)

Official cell-mean BA per arm@variant and mode, mean over orders; paired deltas against `enc@base` (and against a harness
predictions.npz of another system, e.g. cap250, when --ref is given) with 80 % intervals that resample people (cells
stay with their person, orders averaged inside a cell). Labels are opened only here.
"""
import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "harness"))
import data as D  # noqa: E402


def boot(per_person, n_cells):
    people = sorted(per_person)
    v, w = np.array([per_person[p] for p in people]), np.array([n_cells[p] for p in people], float)
    idx = np.random.default_rng(20261120).integers(len(people), size=(10000, len(people)))
    b = (v[idx] * w[idx]).sum(1) / w[idx].sum(1)
    return float((v * w).sum() / w.sum()), [float(q) for q in np.quantile(b, [0.1, 0.9])]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--ref", default=None)
    ap.add_argument("--data", default=None)
    a = ap.parse_args()
    d = D.load(a.data)
    cells = {}  # (mode, arm) -> {(person, session, interface): [ba per order]}
    for f in sorted(Path(a.run).glob("dev_*_*.npz")):
        mode = f.stem.split("_")[1]
        with np.load(f) as z:
            for k in z.files:
                m = re.fullmatch(r"(\w+@[^_]+)_(\d+)_(\w+)", k)
                if not m:
                    continue
                arm, order, ses = m.groups()
                rows = z[f"rows_{order}_{ses}"]
                pred, mask = np.full(len(d["y"]), -1, np.int8), np.zeros(len(d["y"]), bool)
                pred[rows], mask[rows] = z[k], True
                for c in D.cell_table(d, pred, mask, int(order), mode):
                    cells.setdefault((mode, arm), {}).setdefault((c["person"], c["session"], c["interface"]), []).append(c["ba"])
    ref = {}
    if a.ref:
        with np.load(a.ref) as z:
            for k in z.files:
                mode, order, person = k.split("|")
                pred = z[k]
                for c in D.cell_table(d, pred, pred >= 0, int(order), mode):
                    ref.setdefault(mode, {}).setdefault((c["person"], c["session"], c["interface"]), []).append(c["ba"])
    out = {}
    for (mode, arm), tab in sorted(cells.items()):
        ba = {c: float(np.mean(v)) for c, v in tab.items()}
        rec = dict(cell_mean=float(np.mean(list(ba.values()))), n_cells=len(ba), n_orders=len(next(iter(tab.values()))),
                   by_person={p: float(np.mean([v for c, v in ba.items() if c[0] == p])) for p in sorted({c[0] for c in ba})})
        for name, other in (("vs_enc", cells.get((mode, "enc@base"), {})), ("vs_ref", ref.get(mode, {}))):
            if other and set(other) == set(tab) and (name == "vs_ref" or arm != "enc@base"):
                ppl = sorted({c[0] for c in ba})
                dp = {p: float(np.mean([ba[c] - np.mean(other[c]) for c in ba if c[0] == p])) for p in ppl}
                delta, ci = boot(dp, {p: sum(c[0] == p for c in ba) for p in ppl})
                rec[name] = dict(delta=delta, people80=ci, by_person=dp)
        out[f"{mode}|{arm}"] = rec
        extra = "".join(f"  {n} {rec[n]['delta']:+.2f} [{rec[n]['people80'][0]:+.2f}, {rec[n]['people80'][1]:+.2f}]"
                        for n in ("vs_enc", "vs_ref") if n in rec)
        print(f"{mode:3} {arm:28} {rec['cell_mean']:6.2f}  ({rec['n_cells']} cells, {rec['n_orders']} orders){extra}")
    json.dump(out, open(Path(a.run) / "dev_scores.json", "w"), indent=1)


if __name__ == "__main__":
    main()
