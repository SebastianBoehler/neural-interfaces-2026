"""Summarise hardcb(_safe) - stock per order from scores.json files (stdlib + numpy; labels were opened only in score.py).
usage: pkgcmp_collect.py PROD_RUN OV_RUN OUT.md   PROD_RUN holds stock/ and hardcb/ (orders 71000-71003); OV_RUN holds o<first>/{stock,hardcb_safe}/.
Per order: new-people cell-mean delta; returning P04/P10 from by_person. Intervals: order bootstrap and people-nested bootstrap
(people resampled, then orders), 80 %, seed 0, 10,000 draws. Missing chunks are listed, never skipped silently."""
import json
import sys
from pathlib import Path

import numpy as np

prod, ov, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
arms = {}  # order -> (stock scores, hard scores)
for k, (s, h) in {"prod": (prod / "stock", prod / "hardcb")}.items():
    a, b = json.load(open(s / "scores.json")), json.load(open(h / "scores.json"))
    for key in a:
        if key.startswith("new|"):
            arms[int(key[4:])] = (a, b, "hardcb")
missing = []
for d in sorted(ov.glob("o*")):
    s, h = d / "stock" / "scores.json", d / "hardcb_safe" / "scores.json"
    if not (s.exists() and h.exists()):
        missing.append(d.name)
        continue
    a, b = json.load(open(s)), json.load(open(h))
    for key in a:
        if key.startswith("new|"):
            arms[int(key[4:])] = (a, b, "hardcb_safe")
orders = sorted(arms)
rng = np.random.default_rng(0)


def per_person(mode, o, persons):
    a, b, _ = arms[o]
    return np.array([b[f"{mode}|{o}"]["by_person"][p] - a[f"{mode}|{o}"]["by_person"][p] for p in persons])


new_people = [f"P{i:02d}" for i in range(1, 11)]
D = np.stack([per_person("new", o, new_people) for o in orders])  # orders x people
cell = np.array([arms[o][1][f"new|{o}"]["cell_mean"] - arms[o][0][f"new|{o}"]["cell_mean"] for o in orders])
lines = [f"# Overnight package comparison, new-order scores (pp, hardcb arm minus stock)", "",
         f"Orders scored: {len(orders)} ({orders[0]}..{orders[-1]}); chunks missing: {missing or 'none'}. 71000-71003 are the production hardcb "
         "(unpatched), later orders hardcb_safe (guard only).", "", "| order | arm | new people (cell mean) | returning P04 | returning P10 |", "|---|---|---:|---:|---:|"]
for i, o in enumerate(orders):
    r = {p: arms[o][1].get(f"ret|{o}", {}).get("by_person", {}).get(p, np.nan) - arms[o][0].get(f"ret|{o}", {}).get("by_person", {}).get(p, np.nan)
         for p in ("P04", "P10")}
    lines.append(f"| {o} | {arms[o][2]} | {cell[i]:+.2f} | {r['P04']:+.2f} | {r['P10']:+.2f} |")


def boot(M):  # M orders x units; resample units and orders
    n, m = M.shape
    v = [M[rng.integers(0, n, n)][:, rng.integers(0, m, m)].mean() for _ in range(10000)]
    return np.percentile(v, [10, 90])


def order_boot(x):
    return np.percentile([x[rng.integers(0, len(x), len(x))].mean() for _ in range(10000)], [10, 90])


new_ids = [i for i, o in enumerate(orders) if o >= 71004]
lines += ["", "## New people", f"- all {len(orders)} orders: mean {cell.mean():+.3f}, order 80 % [{order_boot(cell)[0]:+.2f}, {order_boot(cell)[1]:+.2f}], "
          f"people-nested 80 % [{boot(D)[0]:+.2f}, {boot(D)[1]:+.2f}]; positive in {int((cell > 0).sum())}/{len(cell)} orders"]
if new_ids:
    c2 = cell[new_ids]
    lines.append(f"- new orders only (71004+, {len(c2)}): mean {c2.mean():+.3f}, order 80 % [{order_boot(c2)[0]:+.2f}, {order_boot(c2)[1]:+.2f}], "
                 f"people-nested [{boot(D[new_ids])[0]:+.2f}, {boot(D[new_ids])[1]:+.2f}]; positive {int((c2 > 0).sum())}/{len(c2)}")
lines.append(f"- per person mean over orders: " + ", ".join(f"{p} {D[:, j].mean():+.2f}" for j, p in enumerate(new_people)))
R = np.array([[arms[o][1][f"ret|{o}"]["by_person"][p] - arms[o][0][f"ret|{o}"]["by_person"][p] for p in ("P04", "P10")]
              for o in orders if f"ret|{o}" in arms[o][0]])
if len(R):
    lines += ["", "## Returning (P04, P10 reported separately; neither decides)",
              f"- P04 mean {R[:, 0].mean():+.2f}, P10 mean {R[:, 1].mean():+.2f}, 0.2/0.8 mix (UNVERIFIED) {0.2 * R[:, 0].mean() + 0.8 * R[:, 1].mean():+.2f}; "
              f"order-bootstrap 80 % P04 {order_boot(R[:, 0]).round(2).tolist()}, P10 {order_boot(R[:, 1]).round(2).tolist()}"]
lines += ["", "Rule (docs/LEDGER.md, Overnight 21 Nov): mean <= 0 on new people refutes; (0,+0.2) is small/undecided. Sealed 57/43 weighting is an UNVERIFIED assumption."]
Path(out).write_text("\n".join(lines) + "\n")
print("\n".join(lines))
