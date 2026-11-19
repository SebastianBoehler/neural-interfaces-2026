"""CBraMod-C2 vs REVE-C (Lane C) from the sealed ensemble banks: BA per order, member BA, error correlation, label-only union."""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent / "laneC")]
from collect import ba  # noqa: E402
from oracle import metadata  # noqa: E402

root, c2, c1 = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
d = metadata(root / "data/proteus_windows.npz")
out = {}
for role in ("held_new", "returning"):
    a, b = np.load(c2 / f"ensemble_C_{role}.npz"), np.load(c1 / f"ensemble_C_{role}.npz")
    rows = a["index"]
    np.testing.assert_array_equal(rows, b["index"])
    y = d["y"][rows]
    r = {}
    for o in (f"full{i}" for i in range(8)):
        np.testing.assert_array_equal(a[o + "_delivery"], b[o + "_delivery"])
        r[o] = dict(c2=ba(d, rows, a[o].argmax(1))[0], reve=ba(d, rows, b[o].argmax(1))[0])
    pa, pb = a["full0"].argmax(1), b["full0"].argmax(1)
    ea, eb = pa != y, pb != y
    members = [ba(d, rows, m.argmax(1))[0] for m in a["full0_members"]]
    rev_members = [ba(d, rows, m.argmax(1))[0] for m in b["full0_members"]]
    w = np.zeros(len(d["y"])); 
    union = (pa == y) | (pb == y)
    conf = (a["full0"].max(1) >= 0.6)
    out[role] = dict(
        c2_mean=float(np.mean([v["c2"] for v in r.values()])), reve_mean=float(np.mean([v["reve"] for v in r.values()])),
        c2_orders=[round(v["c2"], 2) for v in r.values()], reve_orders=[round(v["reve"], 2) for v in r.values()],
        c2_members_full0=[round(m, 2) for m in members], reve_members_full0=[round(m, 2) for m in rev_members],
        error_phi_c2_reve=float(np.corrcoef(ea, eb)[0, 1]),
        member_phi_c2_reve=[float(np.corrcoef(m.argmax(1) != y, eb)[0, 1]) for m in a["full0_members"]],
        union_ba=ba(d, rows, np.where(pb == y, pb, np.where(pa == y, pa, pb)))[0],
        unique_c2_repair_of_reve_misses_share=float(((pa == y) & eb).sum() / max(eb.sum(), 1)),
        confident_unique_repair_share=float(((pa == y) & eb & conf).sum() / max(eb.sum(), 1)),
        confident_regress_share=float(((pa != y) & ~eb & conf).sum() / len(y)))
print(json.dumps(out, indent=1))
(c2 / "compare_c2_vs_reve.json").write_text(json.dumps(out, indent=1) + "\n")
