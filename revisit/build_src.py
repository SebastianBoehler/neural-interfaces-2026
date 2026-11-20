"""Revisit: build src_ORDER.npz (recording-layout clean-slate posteriors + delivery) for revisit/chain/final_r.py.

python revisit/build_src.py ORDER PERMS OUT [--tc-c1 DIR] [--tc DIR] [--tb DIR]
ORDER full0 -> perms key 'full|0|all_loso'; within0 -> 'within|0|all_loso'. Sources (each optional, all ten people required):
  tcA1, tcA2  Team C C1 packages: --tc-c1 = C1 eval dir (order 0 only, keys 'V|0|post'), or --tc = revisit stream_tc outputs
  tbE         Team B R1 encoder ensemble (revisit stream_tb outputs, 'post|KEY')
  tcb         0.5 tcA2 + 0.5 tbE (when both exist)
Every person's streamed rows must equal the perms key restricted to that person (same delivery as the chain)."""
import argparse
from pathlib import Path

import numpy as np

KEYS = {"full0": "full|0|all_loso", "within0": "within|0|all_loso"}
PEOPLE = [f"P{k:02d}" for k in range(1, 11)]


def collect(paths_rows_posts, delivery, n=2655):
    out = np.full((n, 3), np.nan)
    for rows, post in paths_rows_posts:
        np.testing.assert_array_equal(rows, delivery[np.isin(delivery, rows)])
        out[rows] = post
    assert np.isfinite(out).all(), "missing people"
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("order", choices=KEYS)
    ap.add_argument("perms")
    ap.add_argument("out")
    ap.add_argument("--tc-c1")
    ap.add_argument("--tc")
    ap.add_argument("--tb")
    a = ap.parse_args()
    key = KEYS[a.order]
    with np.load(a.perms) as p:
        delivery = p[key]
    src = {}
    for v in ("A1", "A2"):
        if a.tc_c1:
            assert a.order == "full0", "C1 evals hold order 0 = full0 only"
            parts = []
            for q in PEOPLE:
                with np.load(Path(a.tc_c1) / f"eval_{q}.npz") as z:
                    parts.append((z["rows|0"], z[f"{v}|0|post"]))
            src[f"tc{v}"] = collect(parts, delivery)
        elif a.tc:
            tag = key.rsplit("|", 1)[0].replace("|", "_")
            parts = []
            for q in PEOPLE:
                with np.load(Path(a.tc) / f"tc_{tag}_{q}.npz") as z:
                    assert str(z["key"]) == key
                    parts.append((z["rows"], z[f"{v}|post"]))
            src[f"tc{v}"] = collect(parts, delivery)
    if a.tb:
        parts = []
        for q in PEOPLE:
            with np.load(Path(a.tb) / f"tb_{q}.npz") as z:
                parts.append((z[f"rows|{key}"], z[f"post|{key}"]))
        src["tbE"] = collect(parts, delivery)
    if "tcA2" in src and "tbE" in src:
        src["tcb"] = 0.5 * src["tcA2"] + 0.5 * src["tbE"]
    for k in src:
        src[k] = src[k] / src[k].sum(1, keepdims=True)
    np.savez(a.out, order=a.order, delivery=delivery, names=np.array(list(src)), **src)
    print("wrote", a.out, list(src))


if __name__ == "__main__":
    main()
