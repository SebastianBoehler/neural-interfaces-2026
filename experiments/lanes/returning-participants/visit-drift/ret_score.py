"""Lane D3 P04/P10 scoring (local copy of Galvani ret/out). usage: ret_score.py OUT_DIR PROTEUS_WINDOWS.npz DEST.json
cells = order (full0, full1) x person x visit (2, 3); BA3; delta D3f/D3g minus hardcb per cell, per person mean, equal-cell mean; gated minus fixed; drift stats and
paired seconds per session from the receipts."""
import json
import sys
from pathlib import Path

import numpy as np

out, wpath, dest = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
with np.load(wpath, allow_pickle=False) as z:
    Y = z["y"]
ba = lambda y, p: 100 * np.mean([(p[y == c] == c).mean() for c in range(3)])
cell, secs, drift = {}, {}, {}
for d in sorted(out.glob("full*_P*_*")):
    if d.name.endswith("smoke") or not (d / f"{d.name}.npz").exists():
        continue
    with np.load(d / f"{d.name}.npz") as z:
        for a in ("hardcb", "D3f", "D3g"):
            cell[(d.name, a)] = ba(Y[z["rows"]], z[f"{a}|labels"])
    st = json.loads((d / f"{d.name}.json").read_text())["stats"]
    for a in ("hardcb", "D3f", "D3g"):
        secs.setdefault(a, []).append(st[a]["seconds"])
    for a in ("D3f", "D3g"):
        drift.setdefault(a, []).append(st[a]["d3"])
names = sorted({k for k, _ in cell})
res = dict(cells=len(names), table={f"{n}|{a}": v for (n, a), v in cell.items()})
for a in ("D3f", "D3g"):
    dl = {n: cell[(n, a)] - cell[(n, "hardcb")] for n in names}
    res[a] = dict(equal_cell=float(np.mean(list(dl.values()))), **{p: float(np.mean([v for n, v in dl.items() if f"_{p}_" in n])) for p in ("P04", "P10")},
                  cells=dl, drift=dict(active=sum(x["active"] for x in drift[a]), routed=sum(x["routed"] for x in drift[a]),
                                       r_mean=sum(x["r_sum"] for x in drift[a]) / max(sum(x["active"] for x in drift[a]), 1)))
res["gated_minus_fixed_equal_cell"] = res["D3g"]["equal_cell"] - res["D3f"]["equal_cell"]
res["seconds_per_session"] = {a: float(np.mean(v)) for a, v in secs.items()}
Path(dest).write_text(json.dumps(res, indent=1))
print(json.dumps({k: v for k, v in res.items() if k != "table"}, indent=1, default=float))
