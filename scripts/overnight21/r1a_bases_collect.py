"""Collect R1a-on-base results: python r1a_bases_collect.py DATA.npz OUT_DIR  (OUT_DIR as written by r1a_bases.py).

Arms hard, hard_r1a, hardcb, hardcb_r1a; people P04/P10 (visits 2-3). Prints per-order-set tables of paired gains
(equal-cell mean over person x visit x interface cells; paired 80 % intervals resample people, then orders, 20,000
draws, seed 20261121; the 20/80 P04/P10 mix is an UNVERIFIED assumption) and the held-new zero-change summary.
"""
import glob
import json
import sys
from pathlib import Path

import numpy as np

PEOPLE = ("P04", "P10")
ARMS = ("hard", "hard_r1a", "hardcb", "hardcb_r1a")
DRAWS, SEED = 20_000, 20261121


def load(out, arm, person):
    tape = {}
    for f in sorted(glob.glob(str(out / arm / f"pred_ret_{person}_*.npz"))):
        with np.load(f) as t:
            for k in t.files:
                if k in tape:
                    raise ValueError(f"duplicate {k} in {f}")
                tape[k] = t[k]
    return tape


def cell_scores(data, tape):
    y, itf = data["y"], data["interface"]
    out = {}
    for key in tape:
        if key.startswith("rows_"):
            _, order, visit = key.split("_")
            rows, lab = tape[key], tape[f"labels_{order}_{visit}"].astype(int)
            for face in ("Graz", "BrainHero"):
                m = itf[rows] == face
                if set(y[rows][m]) != {0, 1, 2}:
                    raise ValueError(f"incomplete classes {key} {face}")
                out[int(order), visit, face] = 100 * np.mean([(lab[m][y[rows][m] == c] == c).mean() for c in range(3)])
    return out


def interval(g):
    lo, hi = np.quantile(g, (0.1, 0.9))
    return f"[{lo:+.2f}, {hi:+.2f}]"


def nested(gain):
    rng = np.random.default_rng(SEED)
    p, o = gain.shape
    person = [interval(gain[i][rng.integers(o, size=(DRAWS, o))].mean(axis=1)) for i in range(p)]
    draws = gain[rng.integers(p, size=(DRAWS, p, 1)), rng.integers(o, size=(DRAWS, p, o))].mean(axis=(1, 2))
    return person, interval(draws), float(np.mean(draws > 0))


def main(data_path, out):
    out = Path(out)
    with np.load(data_path, allow_pickle=False) as d:
        data = {k: d[k] for k in ("y", "interface")}
    tabs = {}
    for arm in ARMS:
        for person in PEOPLE:
            tape = load(out, arm, person)
            if tape:
                tabs[arm, person] = cell_scores(data, tape)
    pairs = [("hard_r1a", "hard"), ("hardcb_r1a", "hardcb"), ("hard", "hardcb"), ("hard_r1a", "hardcb_r1a")]
    for label, orders in (("orders 71000-71007", range(71000, 71008)), ("orders 71004-71015", range(71004, 71016)),
                          ("orders 71000-71015", range(71000, 71016))):
        print(f"\n### {label}\n\n| Arm − reference | orders | P04 Δ [80%] | P10 Δ [80%] | Equal cells Δ [80%, people-nested] | P(Δ>0) | 20/80 Δ* |\n|---|---:|---|---|---|---:|---:|")
        for a, b in pairs:
            if not all((x, p) in tabs for x in (a, b) for p in PEOPLE):
                continue
            have = [o for o in orders if all(any(k[0] == o for k in tabs[x, p]) for x in (a, b) for p in PEOPLE)]
            if len(have) < 2:
                continue
            gain = np.array([[np.mean([tabs[a, p][o, v, f] - tabs[b, p][o, v, f] for v in ("2", "3") for f in ("Graz", "BrainHero")])
                              for o in have] for p in PEOPLE])
            ci, nci, pos = nested(gain)
            m = gain.mean(axis=1)
            print(f"| {a} − {b} | {len(have)} | {m[0]:+.2f} {ci[0]} | {m[1]:+.2f} {ci[1]} | {gain.mean():+.2f} {nci} | {pos:.2f} | {m @ [0.2, 0.8]:+.2f} |")
    print("\n### Absolute equal-cell BA (all collected orders)")
    for arm in ARMS:
        if all((arm, p) in tabs for p in PEOPLE):
            print(f"- {arm}: " + ", ".join(f"{p} {np.mean(list(tabs[arm, p].values())):.2f}" for p in PEOPLE))
    print("\n### Run checks (info files)")
    for f in sorted(glob.glob(str(out / "info_*.json"))):
        rows = json.load(open(f))
        name = Path(f).stem
        miss = sum(r.get("stored_mismatch_hardcb", 0) + r.get("stored_mismatch_hardcb_r1a", 0) for r in rows)
        print(f"- {name}: {len(rows)} streams, recognised {sorted({r['recognised'] for r in rows})}, late {sorted({tuple(r['late']) for r in rows})}, "
              f"labels changed {sum(r['changed'] for r in rows)}, stored-tape mismatches {miss if 'stored_mismatch_hardcb' in rows[0] else 'n/a'}")


if __name__ == "__main__":
    main(*sys.argv[1:3])
