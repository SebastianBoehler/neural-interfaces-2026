"""Lane G1 analysis (numpy only; cells.json from chain/score_g1.py). Rule in evidence/wave_20nov/laneG1_emgraph.md.
usage: python analyse_g1.py cells.json"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "laneS2"))
from analyse_s2 import ORDERS, ba, contrast, contrast_po, parse, person_delta  # noqa: E402

PAIRS = [("anchor", "base"), ("shared", "base"), ("cog", "base"), ("wrongWM", "base"), ("wrongSM", "base"),
         ("cog", "anchor"), ("wrongWM", "anchor"), ("shared", "anchor"), ("cog", "wrongWM"), ("cog", "wrongSM"), ("cog", "shared")]


def main(path):
    raw = json.loads(Path(path).read_text())
    c = parse(path)
    draw = np.random.default_rng(460).integers(0, 10, (10000, 10))
    f = lambda x: f"{x[0]:+.2f} [{x[1][0]:+.2f},{x[1][1]:+.2f}]"
    people = [p.rsplit("/", 1)[-1] for p in c["full0"]["hardcb|base"][2]]
    new = np.array([p not in ("P04", "P10") for p in people])
    for sy in ("hardcb", "ds_all"):
        z = np.ones(10, bool)
        print(f"\n## {sy}: base both {np.mean([ba(c, o, sy + '|base', z) for o in ORDERS]):.2f} (full0 {ba(c, 'full0', sy + '|base', z):.2f} / within0 {ba(c, 'within0', sy + '|base', z):.2f}); zero == base: "
              f"{all(c[o][sy + '|zero'][0].tolist() == c[o][sy + '|base'][0].tolist() for o in ORDERS) if sy + '|zero' in c['full0'] else 'n/a'}")
        print("pair | both [people 80%] (full0/within0) [people+orders 80%] | P04 P10 | J UNVERIFIED 57/43 mix, 0.2 P04 + 0.8 P10 (pp)")
        for a, b in PAIRS:
            A, B = f"{sy}|{a}", f"{sy}|{b}"
            r = contrast(c, A, B, ORDERS, draw)
            f0, w0 = contrast(c, A, B, ("full0",), draw)[0], contrast(c, A, B, ("within0",), draw)[0]
            d = np.mean([[100 * (c[o][A][0][i] - c[o][B][0][i]) / c[o][A][1][i] for i in range(10)] for o in ORDERS], 0)
            J = 0.57 * d[new].mean() + 0.43 * (0.2 * d[people.index("P04")] + 0.8 * d[people.index("P10")])
            print(f"{a:8s} - {b:7s} {f(r)} ({f0:+.2f}/{w0:+.2f}) [{contrast_po(c, A, B)[0]:+.2f},{contrast_po(c, A, B)[1]:+.2f}] | {d[people.index('P04')]:+.2f} {d[people.index('P10')]:+.2f} | {J:+.2f}")
    m = raw["minority_recall_delta"]
    print("\n## minority-class recall delta vs base, ages 20-60 (pp; mean over people, both orders; people listed with a minority session only)")
    for sy in ("hardcb", "ds_all"):
        for a in ("anchor", "shared", "cog", "wrongWM", "wrongSM"):
            v = np.nanmean([[m[f"{o}|{sy}|{a}"][p] for p in m[f"{o}|{sy}|{a}"]] for o in ORDERS], 0)
            print(f"{sy:7s} {a:8s} {100*np.nanmean(v):+.2f} (people 80% [{np.percentile(100*np.nanmean(v[draw], 1), 10):+.2f},{np.percentile(100*np.nanmean(v[draw], 1), 90):+.2f}])")


if __name__ == "__main__":
    main(sys.argv[1])
